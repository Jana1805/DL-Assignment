/**
 * Compares the metrics this extension computes with the values recorded in
 * the ApacheJIT dataset, for commits of one cloned Apache project.
 *
 * Usage:
 *   npm run calibrate -- <repo path> <apachejit_total.csv> <project> [sample size]
 *   npm run calibrate -- ../scratch/zookeeper ../dataset/apachejit_total.csv apache/zookeeper 200
 */

import { readFileSync } from 'node:fs';
import { CommitMetrics, computeMetrics } from './metrics';

const METRICS: (keyof CommitMetrics)[] = [
    'la', 'ld', 'nf', 'nd', 'ns', 'ent', 'ndev', 'age', 'nuc', 'aexp', 'arexp', 'asexp',
];

function median(values: number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function correlation(xs: number[], ys: number[]): number {
    const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
    const mx = mean(xs);
    const my = mean(ys);
    let cov = 0;
    let vx = 0;
    let vy = 0;
    for (let i = 0; i < xs.length; i++) {
        cov += (xs[i] - mx) * (ys[i] - my);
        vx += (xs[i] - mx) ** 2;
        vy += (ys[i] - my) ** 2;
    }
    return vx > 0 && vy > 0 ? cov / Math.sqrt(vx * vy) : NaN;
}

async function main(): Promise<void> {
    const [repo, csvPath, project, sizeArg] = process.argv.slice(2);
    if (!repo || !csvPath || !project) {
        throw new Error('Usage: calibrate <repo path> <apachejit_total.csv> <project> [sample size]');
    }
    const sampleSize = Number(sizeArg ?? 200);

    const [headerLine, ...lines] = readFileSync(csvPath, 'utf8').trim().split(/\r?\n/);
    const header = headerLine.split(',');
    const rows = lines
        .map((line) => line.split(','))
        .filter((cells) => cells[header.indexOf('project')] === project);
    const step = Math.max(1, Math.floor(rows.length / sampleSize));
    const sample = rows.filter((_, index) => index % step === 0).slice(0, sampleSize);

    const expected: Record<string, number[]> = {};
    const actual: Record<string, number[]> = {};
    METRICS.forEach((m) => { expected[m] = []; actual[m] = []; });

    let missing = 0;
    for (const cells of sample) {
        const rev = cells[header.indexOf('commit_id')];
        const metrics = await computeMetrics(repo, { kind: 'commit', rev }).catch(() => undefined);
        if (!metrics) {
            missing += 1;
            continue;
        }
        for (const m of METRICS) {
            expected[m].push(Number(cells[header.indexOf(m)]));
            actual[m].push(metrics[m]);
        }
    }

    console.log(`${project}: compared ${sample.length - missing} commits (${missing} not found)\n`);
    console.log('metric  exact%  within10%  log-corr  median(dataset)  median(ours)');
    for (const m of METRICS) {
        const e = expected[m];
        const a = actual[m];
        const close = (tolerance: number) =>
            e.filter((value, i) => Math.abs(a[i] - value) <= tolerance * Math.max(Math.abs(value), 1e-9) + 1e-6).length;
        console.log([
            m.padEnd(6),
            ((100 * close(0.001)) / e.length).toFixed(0).padStart(6),
            ((100 * close(0.1)) / e.length).toFixed(0).padStart(9),
            correlation(e.map(Math.log1p), a.map(Math.log1p)).toFixed(3).padStart(9),
            median(e).toFixed(2).padStart(16),
            median(a).toFixed(2).padStart(13),
        ].join('  '));
    }
}

main().catch((error) => {
    console.error(error);
    process.exit(1);
});
