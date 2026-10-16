import * as vscode from 'vscode';
import { predict } from './client';
import { computeMetrics, git, Target } from './metrics';
import { NotifyMode, RiskStatus } from './ui';

/** The parts of the built-in Git extension API this extension uses. */
interface GitRepository {
    rootUri: vscode.Uri;
    state: {
        HEAD?: { commit?: string };
        onDidChange: vscode.Event<void>;
    };
}

interface GitApi {
    repositories: GitRepository[];
    onDidOpenRepository: vscode.Event<GitRepository>;
}

let status: RiskStatus;

async function score(cwd: string, target: Target, label: string): Promise<void> {
    const config = vscode.workspace.getConfiguration('jitRisk');
    status.working();
    try {
        const metrics = await computeMetrics(cwd, target);
        if (!metrics) {
            status.unavailable('no changed files');
            return;
        }
        const prediction = await predict(metrics, {
            apiUrl: config.get<string>('apiUrl', ''),
            apiKey: config.get<string>('apiKey', ''),
            timeoutMs: config.get<number>('timeoutMs', 8000),
        });
        status.show(label, metrics, prediction, config.get<NotifyMode>('notify', 'risky'));
    } catch (error) {
        // Scoring is best-effort: never interrupt the user's commit flow.
        status.unavailable(error instanceof Error ? error.message : String(error));
    }
}

/** True when the latest HEAD movement was a commit, not a checkout or pull. */
async function headMovedByCommit(cwd: string): Promise<boolean> {
    try {
        return (await git(cwd, ['reflog', '-1', '--format=%gs'])).startsWith('commit');
    } catch {
        return false;
    }
}

function watch(repository: GitRepository, context: vscode.ExtensionContext): void {
    let lastHead = repository.state.HEAD?.commit;
    context.subscriptions.push(
        repository.state.onDidChange(async () => {
            const head = repository.state.HEAD?.commit;
            if (!head || head === lastHead) {
                return;
            }
            lastHead = head;
            const cwd = repository.rootUri.fsPath;
            if (await headMovedByCommit(cwd)) {
                await score(cwd, { kind: 'commit', rev: head }, head.slice(0, 7));
            }
        }),
    );
}

function activeRepository(api: GitApi): GitRepository | undefined {
    const active = vscode.window.activeTextEditor?.document.uri;
    const containing = active && api.repositories.find((r) => active.fsPath.startsWith(r.rootUri.fsPath));
    return containing ?? api.repositories[0];
}

export async function activate(context: vscode.ExtensionContext): Promise<void> {
    status = new RiskStatus();
    context.subscriptions.push(status);

    const gitExtension = vscode.extensions.getExtension('vscode.git');
    if (!gitExtension) {
        status.unavailable('the built-in Git extension is disabled');
        return;
    }
    const api: GitApi = (await gitExtension.activate()).getAPI(1);

    api.repositories.forEach((repository) => watch(repository, context));
    context.subscriptions.push(api.onDidOpenRepository((repository) => watch(repository, context)));

    const command = (name: string, target: (r: GitRepository) => Target, label: string) =>
        vscode.commands.registerCommand(name, async () => {
            const repository = activeRepository(api);
            if (!repository) {
                void vscode.window.showInformationMessage('Commit risk: no Git repository is open.');
                return;
            }
            await score(repository.rootUri.fsPath, target(repository), label);
        });

    context.subscriptions.push(
        command('jitRisk.scoreStaged', () => ({ kind: 'staged' }), 'staged changes'),
        command('jitRisk.scoreLastCommit', () => ({ kind: 'commit', rev: 'HEAD' }), 'last commit'),
    );
}

export function deactivate(): void {}
