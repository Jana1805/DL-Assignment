/**
 * Computes the 12 ApacheJIT change metrics for a commit (or the staged
 * changes) by running git. No dependency on the VS Code API, so the same
 * code runs in tests and in the calibration script.
 */

import { execFile } from 'node:child_process';

export interface CommitMetrics {
    la: number;
    ld: number;
    nf: number;
    nd: number;
    ns: number;
    ent: number;
    ndev: number;
    age: number;
    nuc: number;
    aexp: number;
    arexp: number;
    asexp: number;
}

export interface FileChange {
    path: string;
    added: number;
    deleted: number;
}

export interface HistoryCommit {
    author: string;
    time: number;
    files: string[];
}

export type Target = { kind: 'commit'; rev: string } | { kind: 'staged' };

const SECONDS_PER_DAY = 86400;
const SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY;
const LOG_FORMAT = '--format=%x01%ae%x09%at';
const MAX_PATHSPEC_CHARS = 6000;

export function git(cwd: string, args: string[], timeoutMs = 30000): Promise<string> {
    return new Promise((resolve, reject) => {
        execFile(
            'git',
            ['-c', 'core.quotepath=off', ...args],
            {
                cwd,
                timeout: timeoutMs,
                maxBuffer: 256 * 1024 * 1024,
                env: { ...process.env, GIT_LITERAL_PATHSPECS: '1' },
            },
            (error, stdout) => (error ? reject(error) : resolve(stdout)),
        );
    });
}

/** Parses `--numstat -z --no-renames` output. Binary files count as 0 lines. */
export function parseNumstat(output: string): FileChange[] {
    const changes: FileChange[] = [];
    for (const record of output.split('\0')) {
        const [added, deleted, ...rest] = record.replace(/^\n+/, '').split('\t');
        if (rest.length === 0) {
            continue;
        }
        changes.push({
            path: rest.join('\t'),
            added: added === '-' ? 0 : Number(added),
            deleted: deleted === '-' ? 0 : Number(deleted),
        });
    }
    return changes;
}

/** Parses `git log --format=%x01%ae%x09%at --name-only` output. */
export function parseLog(output: string): HistoryCommit[] {
    const commits: HistoryCommit[] = [];
    for (const block of output.split('\x01').slice(1)) {
        const [header, ...lines] = block.split('\n');
        const [author, time] = header.split('\t');
        commits.push({
            author: author.toLowerCase(),
            time: Number(time),
            files: lines.filter((line) => line.length > 0),
        });
    }
    return commits;
}

// The definitions below follow what the ApacheJIT dataset actually contains,
// checked against real Apache commits with `npm run calibrate`. Where that
// differs from the textbook metric, the dataset wins: the model only knows
// the dataset's version.

/** The name of the file's parent directory (not its full path). */
export function directoryOf(path: string): string {
    const parts = path.split('/');
    return parts.length === 1 ? parts[0] : parts[parts.length - 2];
}

/** The first path component, so each root-level file is its own subsystem. */
export function subsystemOf(path: string): string {
    return path.split('/')[0];
}

/** Size and diffusion metrics, from the diff alone. */
export function diffMetrics(changes: FileChange[]): Pick<CommitMetrics, 'la' | 'ld' | 'nf' | 'nd' | 'ns' | 'ent'> {
    const la = changes.reduce((sum, c) => sum + c.added, 0);
    const ld = changes.reduce((sum, c) => sum + c.deleted, 0);
    const total = la + ld;

    let ent = 0;
    for (const change of changes) {
        const share = total > 0 ? (change.added + change.deleted) / total : 0;
        if (share > 0) {
            ent -= share * Math.log2(share);
        }
    }

    return {
        la,
        ld,
        nf: changes.length,
        nd: new Set(changes.map((c) => directoryOf(c.path))).size,
        ns: new Set(changes.map((c) => subsystemOf(c.path))).size,
        ent,
    };
}

/**
 * History metrics, each divided by the number of touched files: distinct
 * earlier developers across all files, whole days since each file's last
 * change, and earlier changes per file.
 */
export function historyMetrics(
    paths: string[],
    history: HistoryCommit[],
    now: number,
): Pick<CommitMetrics, 'ndev' | 'age' | 'nuc'> {
    const perFile = new Map(paths.map((p) => [p, { changes: 0, last: 0 }]));
    const developers = new Set<string>();
    for (const commit of history) {
        for (const file of commit.files) {
            const entry = perFile.get(file);
            if (entry) {
                developers.add(commit.author);
                entry.changes += 1;
                entry.last = Math.max(entry.last, commit.time);
            }
        }
    }

    let nuc = 0;
    let age = 0;
    for (const entry of perFile.values()) {
        nuc += entry.changes;
        age += entry.changes > 0 ? Math.floor(Math.max(0, now - entry.last) / SECONDS_PER_DAY) : 0;
    }
    const count = Math.max(1, perFile.size);
    return { ndev: developers.size / count, age: age / count, nuc: nuc / count };
}

/**
 * In the dataset, `asexp` is the author's earlier commits in the subsystem
 * only when a commit touches one subsystem. For commits touching several
 * (26% of rows) it is an unrelated value between 0 and 2 that cannot be
 * reproduced from git, so the dataset median for those rows is sent instead
 * to keep the model's input in the range it was trained on.
 */
export const MULTI_SUBSYSTEM_ASEXP = 0.5;

/** Author experience from the author's earlier commits. */
export function experienceMetrics(
    paths: string[],
    authorHistory: HistoryCommit[],
    now: number,
): Pick<CommitMetrics, 'aexp' | 'arexp' | 'asexp'> {
    const subsystems = new Set(paths.map(subsystemOf));
    let arexp = 0;
    let asexp = 0;
    for (const commit of authorHistory) {
        const yearsAgo = Math.floor(Math.max(0, now - commit.time) / SECONDS_PER_YEAR);
        arexp += 1 / (yearsAgo + 1);
        if (commit.files.some((file) => subsystems.has(subsystemOf(file)))) {
            asexp += 1;
        }
    }
    return {
        aexp: authorHistory.length,
        arexp,
        asexp: subsystems.size > 1 ? MULTI_SUBSYSTEM_ASEXP : asexp,
    };
}

function chunkPaths(paths: string[]): string[][] {
    const chunks: string[][] = [];
    let current: string[] = [];
    let size = 0;
    for (const path of paths) {
        if (current.length > 0 && size + path.length > MAX_PATHSPEC_CHARS) {
            chunks.push(current);
            current = [];
            size = 0;
        }
        current.push(path);
        size += path.length + 1;
    }
    if (current.length > 0) {
        chunks.push(current);
    }
    return chunks;
}

async function resolves(cwd: string, rev: string): Promise<boolean> {
    try {
        await git(cwd, ['rev-parse', '--verify', '--quiet', `${rev}^{commit}`]);
        return true;
    } catch {
        return false;
    }
}

/**
 * Returns undefined when there is nothing to score (no changed files).
 */
export async function computeMetrics(cwd: string, target: Target): Promise<CommitMetrics | undefined> {
    let changes: FileChange[];
    let author: string;
    let now: number;
    let base: string;

    if (target.kind === 'commit') {
        changes = parseNumstat(
            await git(cwd, ['show', '--numstat', '-z', '--no-renames', '--format=', target.rev]),
        );
        const [email, time] = (await git(cwd, ['show', '-s', '--format=%ae%x09%at', target.rev])).trim().split('\t');
        author = email;
        now = Number(time);
        base = `${target.rev}^`;
    } else {
        changes = parseNumstat(await git(cwd, ['diff', '--cached', '--numstat', '-z', '--no-renames']));
        author = (await git(cwd, ['config', 'user.email']).catch(() => '')).trim();
        now = Date.now() / 1000;
        base = 'HEAD';
    }

    if (changes.length === 0) {
        return undefined;
    }
    const paths = changes.map((c) => c.path);

    let history: HistoryCommit[] = [];
    let authorHistory: HistoryCommit[] = [];
    if (await resolves(cwd, base)) {
        for (const chunk of chunkPaths(paths)) {
            history = history.concat(
                parseLog(await git(cwd, ['log', LOG_FORMAT, '--name-only', base, '--', ...chunk])),
            );
        }
        if (author) {
            authorHistory = parseLog(
                await git(cwd, [
                    'log', LOG_FORMAT, '--name-only', '--fixed-strings', '--regexp-ignore-case',
                    `--author=<${author}>`, base,
                ]),
            );
        }
    }

    return {
        ...diffMetrics(changes),
        ...historyMetrics(paths, history, now),
        ...experienceMetrics(paths, authorHistory, now),
    };
}
