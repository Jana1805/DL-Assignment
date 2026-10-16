import { CommitMetrics } from './metrics';

export interface Prediction {
    risk_score: number;
    is_risky: boolean;
    threshold: number;
    model_version: string;
}

export interface ClientOptions {
    apiUrl: string;
    apiKey?: string;
    timeoutMs: number;
}

/** The API takes whole numbers for the size metrics. */
export function toPayload(metrics: CommitMetrics): CommitMetrics {
    return {
        ...metrics,
        la: Math.round(metrics.la),
        ld: Math.round(metrics.ld),
        nf: Math.round(metrics.nf),
        nd: Math.round(metrics.nd),
        ns: Math.round(metrics.ns),
    };
}

export async function predict(metrics: CommitMetrics, options: ClientOptions): Promise<Prediction> {
    const headers: Record<string, string> = { 'Content-Type': 'application/json' };
    if (options.apiKey) {
        headers['X-API-Key'] = options.apiKey;
    }

    const response = await fetch(`${options.apiUrl.replace(/\/+$/, '')}/predict`, {
        method: 'POST',
        headers,
        body: JSON.stringify(toPayload(metrics)),
        signal: AbortSignal.timeout(options.timeoutMs),
    });
    if (!response.ok) {
        throw new Error(`Risk API returned ${response.status}`);
    }
    return (await response.json()) as Prediction;
}
