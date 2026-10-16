import * as vscode from 'vscode';
import { Prediction } from './client';
import { CommitMetrics } from './metrics';

export type NotifyMode = 'risky' | 'always' | 'never';

export class RiskStatus {
    private readonly item: vscode.StatusBarItem;

    constructor() {
        this.item = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 0);
        this.item.command = 'jitRisk.scoreStaged';
        this.idle();
        this.item.show();
    }

    idle(): void {
        this.item.text = '$(shield) Risk';
        this.item.tooltip = 'Commit risk: click to score the staged changes';
        this.item.backgroundColor = undefined;
    }

    working(): void {
        this.item.text = '$(sync~spin) Risk';
    }

    unavailable(reason: string): void {
        this.item.text = '$(shield) Risk n/a';
        this.item.tooltip = `Commit risk unavailable: ${reason}`;
        this.item.backgroundColor = undefined;
    }

    show(label: string, metrics: CommitMetrics, prediction: Prediction, notify: NotifyMode): void {
        const percent = Math.round(prediction.risk_score * 100);
        this.item.text = `$(shield) Risk ${percent}%`;
        this.item.backgroundColor = prediction.is_risky
            ? new vscode.ThemeColor('statusBarItem.warningBackground')
            : undefined;

        const tooltip = new vscode.MarkdownString();
        tooltip.appendMarkdown(`**Commit risk: ${percent}%** (${label})\n\n`);
        tooltip.appendMarkdown(
            `Flagged at ${Math.round(prediction.threshold * 100)}% or above. ` +
            'A review hint, not a verdict: most flagged commits are fine.\n\n',
        );
        tooltip.appendMarkdown(
            `+${metrics.la} / -${metrics.ld} lines in ${metrics.nf} files, ` +
            `${metrics.nd} directories, ${metrics.ns} subsystems\n\n`,
        );
        tooltip.appendMarkdown(`Model ${prediction.model_version}`);
        this.item.tooltip = tooltip;

        const message = `Commit risk ${percent}% (${label}).`;
        if (prediction.is_risky && notify !== 'never') {
            void vscode.window.showWarningMessage(`${message} Worth a second look before it merges.`);
        } else if (notify === 'always') {
            void vscode.window.showInformationMessage(message);
        }
    }

    dispose(): void {
        this.item.dispose();
    }
}
