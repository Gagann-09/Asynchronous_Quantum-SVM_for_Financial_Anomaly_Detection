import { useState, useRef, useCallback } from 'react';
import axios from 'axios';

type PredictionState = 'idle' | 'submitting' | 'polling_backend' | 'success' | 'error';

interface PredictionResult {
  prediction: string;          // "Normal" | "Anomaly"
  confidence_score: number;    // raw abs(decision_function)
  confidence_pct: number;      // normalized 0–100
  features: number[];
}

/**
 * Sigmoid normalization: maps the unbounded SVM decision function
 * distance to a bounded 0–100% confidence percentage.
 *
 * Formula: confidence_pct = (1 − e^(−k × raw)) × 100
 *   k = 2.0 (steepness constant)
 *   raw = 0.0  →  ~0%   (on the hyperplane, maximum uncertainty)
 *   raw = 1.0  →  ~86%
 *   raw ≥ 2.5  →  ~99%
 */
function normalizeConfidence(raw: number): number {
  const k = 2.0;
  return (1 - Math.exp(-k * Math.abs(raw))) * 100;
}

export function usePrediction() {
  const [state, setState] = useState<PredictionState>('idle');
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const clearPolling = useCallback(() => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
  }, []);

  const submit = useCallback(async (features: number[]) => {
    if (features.length !== 20) {
      setErrorMsg('Exactly 20 features are required.');
      setState('error');
      return;
    }

    setState('submitting');
    setResult(null);
    setErrorMsg(null);
    clearPolling();

    try {
      const response = await axios.post('http://127.0.0.1:8000/api/v1/predict', { features });
      const taskId = response.data.task_id;

      if (!taskId) {
        throw new Error('No task ID returned from backend.');
      }

      setState('polling_backend');

      pollingIntervalRef.current = setInterval(async () => {
        try {
          const statusRes = await axios.get(`http://127.0.0.1:8000/api/v1/status/${taskId}`);
          const { status, result: backendResult } = statusRes.data;

          if (status === 'SUCCESS') {
            clearPolling();
            setResult({
              prediction: backendResult.prediction,
              confidence_score: backendResult.confidence_score,
              confidence_pct: normalizeConfidence(backendResult.confidence_score),
              features
            });
            setState('success');
          } else if (status === 'FAILURE' || status === 'ERROR') {
            clearPolling();
            setErrorMsg('Backend processing failed.');
            setState('error');
          }
          // Otherwise still PENDING/PROCESSING — keep polling
        } catch (pollErr: any) {
          clearPolling();
          setErrorMsg(pollErr.message || 'Error occurred while polling the backend.');
          setState('error');
        }
      }, 2000);

    } catch (submitErr: any) {
      clearPolling();
      setErrorMsg(submitErr.message || 'Failed to submit features to the backend.');
      setState('error');
    }
  }, [clearPolling]);

  return { state, result, errorMsg, submit, clearPolling };
}
