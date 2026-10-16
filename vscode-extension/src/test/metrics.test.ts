import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { after, before, describe, it } from 'node:test';
import { toPayload } from '../client';
import {
    computeMetrics, diffMetrics, experienceMetrics, historyMetrics, MULTI_SUBSYSTEM_ASEXP, parseLog, parseNumstat,
} from '../metrics';

const DAY = 86400;
const YEAR = 365.25 * DAY;

describe('parseNumstat', () => {
    it('reads added, deleted and path from -z output', () => {
        assert.deepEqual(parseNumstat('3\t1\tsrc/a.ts\x0010\t0\tdocs/read me.md\x00'), [
            { path: 'src/a.ts', added: 3, deleted: 1 },
            { path: 'docs/read me.md', added: 10, deleted: 0 },
        ]);
    });

    it('counts binary files as zero lines', () => {
        assert.deepEqual(parseNumstat('-\t-\tlogo.png\x00'), [{ path: 'logo.png', added: 0, deleted: 0 }]);
    });

    it('returns nothing for an empty diff', () => {
        assert.deepEqual(parseNumstat(''), []);
    });
});

describe('diffMetrics', () => {
    it('matches the dataset: 2 files with equal churn have entropy 1', () => {
        const metrics = diffMetrics([
            { path: 'core/a.java', added: 1, deleted: 1 },
            { path: 'core/b.java', added: 1, deleted: 1 },
        ]);
        assert.deepEqual(metrics, { la: 2, ld: 2, nf: 2, nd: 1, ns: 1, ent: 1 });
    });

    it('counts directories and top-level subsystems separately', () => {
        const metrics = diffMetrics([
            { path: 'core/x/a.java', added: 4, deleted: 0 },
            { path: 'core/y/b.java', added: 0, deleted: 0 },
            { path: 'web/c.java', added: 0, deleted: 0 },
            { path: 'README.md', added: 0, deleted: 0 },
        ]);
        assert.equal(metrics.nf, 4);
        assert.equal(metrics.nd, 4);
        assert.equal(metrics.ns, 3);
        assert.equal(metrics.ent, 0);
    });

    it('identifies a directory by its name, as the dataset does', () => {
        const metrics = diffMetrics([
            { path: 'src/main/server/quorum/Peer.java', added: 1, deleted: 0 },
            { path: 'src/test/server/quorum/PeerTest.java', added: 1, deleted: 0 },
            { path: 'CHANGES.txt', added: 1, deleted: 0 },
            { path: 'NOTICE.txt', added: 1, deleted: 0 },
        ]);
        assert.equal(metrics.nd, 3); // quorum, CHANGES.txt, NOTICE.txt
        assert.equal(metrics.ns, 3); // src, CHANGES.txt, NOTICE.txt
    });

    it('has zero entropy when no lines changed', () => {
        assert.equal(diffMetrics([{ path: 'logo.png', added: 0, deleted: 0 }]).ent, 0);
    });
});

describe('historyMetrics', () => {
    it('divides distinct developers, changes and whole days by the file count', () => {
        const now = 100 * DAY;
        const history = parseLog(
            '\x01A@x.org\t' + 90 * DAY + '\n\na.java\nb.java\n' +
            '\x01b@x.org\t' + 95.5 * DAY + '\n\na.java\n',
        );
        const metrics = historyMetrics(['a.java', 'b.java', 'new.java'], history, now);

        assert.equal(metrics.ndev, 2 / 3); // a and b, counted once across files
        assert.equal(metrics.nuc, (2 + 1 + 0) / 3);
        assert.equal(metrics.age, (4 + 10 + 0) / 3); // 4.5 days counts as 4
    });
});

describe('experienceMetrics', () => {
    it('weights recent commits higher and counts subsystem matches', () => {
        const now = 10 * YEAR;
        const history = [
            { author: 'a', time: now - DAY, files: ['core/a.java'] },
            { author: 'a', time: now - 1.5 * YEAR, files: ['web/b.java'] },
            { author: 'a', time: now - 3.2 * YEAR, files: ['core/c.java', 'docs/d.md'] },
        ];
        const metrics = experienceMetrics(['core/z.java'], history, now);

        assert.equal(metrics.aexp, 3);
        assert.equal(metrics.arexp, 1 + 1 / 2 + 1 / 4);
        assert.equal(metrics.asexp, 2);
    });

    it('sends the dataset median for commits touching several subsystems', () => {
        const history = [{ author: 'a', time: 0, files: ['core/a.java'] }];
        const metrics = experienceMetrics(['core/z.java', 'web/y.java'], history, DAY);

        assert.equal(metrics.aexp, 1);
        assert.equal(metrics.asexp, MULTI_SUBSYSTEM_ASEXP);
    });
});

describe('toPayload', () => {
    it('keeps the 12 fields the API accepts', () => {
        const payload = toPayload({
            la: 1, ld: 2, nf: 1, nd: 1, ns: 1, ent: 0, ndev: 0.5, age: 1.5, nuc: 2, aexp: 3, arexp: 2.5, asexp: 1,
        });
        assert.deepEqual(Object.keys(payload).sort(), [
            'aexp', 'age', 'arexp', 'asexp', 'ent', 'la', 'ld', 'nd', 'ndev', 'nf', 'ns', 'nuc',
        ]);
    });
});

describe('computeMetrics on a real repository', () => {
    let repo: string;
    let clock = 1_600_000_000;

    const run = (args: string[], author = 'alice') =>
        execFileSync('git', args, {
            cwd: repo,
            env: {
                ...process.env,
                GIT_AUTHOR_NAME: author,
                GIT_AUTHOR_EMAIL: `${author}@example.org`,
                GIT_COMMITTER_NAME: author,
                GIT_COMMITTER_EMAIL: `${author}@example.org`,
                GIT_AUTHOR_DATE: `${clock} +0000`,
                GIT_COMMITTER_DATE: `${clock} +0000`,
            },
        });

    const write = (path: string, lines: number) => {
        mkdirSync(dirname(join(repo, path)), { recursive: true });
        writeFileSync(join(repo, path), Array.from({ length: lines }, (_, i) => `line ${i}`).join('\n') + '\n');
    };

    const commit = (author: string, days: number) => {
        clock += days * DAY;
        run(['add', '-A'], author);
        run(['commit', '-q', '-m', 'change'], author);
    };

    before(() => {
        repo = mkdtempSync(join(tmpdir(), 'jit-risk-'));
        run(['init', '-q']);
        run(['config', 'user.email', 'alice@example.org']);
        run(['config', 'user.name', 'alice']);

        write('core/a.txt', 10);
        commit('alice', 0);
        write('core/a.txt', 12);
        write('web/b.txt', 5);
        commit('bob', 2);
        write('core/a.txt', 14);
        write('core/c.txt', 2);
        commit('alice', 4);
    });

    after(() => rmSync(repo, { recursive: true, force: true }));

    it('scores the root commit with no history', async () => {
        const root = execFileSync('git', ['rev-list', '--max-parents=0', 'HEAD'], { cwd: repo }).toString().trim();
        const metrics = await computeMetrics(repo, { kind: 'commit', rev: root });

        assert.deepEqual(metrics, {
            la: 10, ld: 0, nf: 1, nd: 1, ns: 1, ent: 0,
            ndev: 0, age: 0, nuc: 0, aexp: 0, arexp: 0, asexp: 0,
        });
    });

    it('scores the latest commit from the history before it', async () => {
        const metrics = await computeMetrics(repo, { kind: 'commit', rev: 'HEAD' });

        assert.ok(metrics);
        assert.equal(metrics.la, 4);
        assert.equal(metrics.ld, 0);
        assert.equal(metrics.nf, 2);
        assert.equal(metrics.nd, 1);
        assert.equal(metrics.ns, 1);
        assert.equal(metrics.ent, 1);
        assert.equal(metrics.ndev, (2 + 0) / 2); // a.txt: alice and bob; c.txt is new
        assert.equal(metrics.nuc, (2 + 0) / 2);
        assert.equal(metrics.age, (4 + 0) / 2);
        assert.equal(metrics.aexp, 1); // alice's root commit only
        assert.equal(metrics.arexp, 1);
        assert.equal(metrics.asexp, 1);
    });

    it('scores staged changes against HEAD', async () => {
        write('web/b.txt', 8);
        run(['add', '-A']);
        const metrics = await computeMetrics(repo, { kind: 'staged' });

        assert.ok(metrics);
        assert.equal(metrics.la, 3);
        assert.equal(metrics.nf, 1);
        assert.equal(metrics.ndev, 1); // bob
        assert.equal(metrics.aexp, 2); // alice has two commits
        assert.equal(metrics.asexp, 0); // none of them in web/
    });

    it('returns undefined when nothing is staged', async () => {
        run(['reset', '-q', '--hard']);
        assert.equal(await computeMetrics(repo, { kind: 'staged' }), undefined);
    });
});
