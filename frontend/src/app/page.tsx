'use client';

import { useState } from 'react';
import { usePrediction } from '@/hooks/usePrediction';
import { Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Progress } from '@/components/ui/progress';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { AlertCircle, CheckCircle2, Loader2, Wand2, ShieldAlert, ShieldCheck } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

export default function Dashboard() {
  const [features, setFeatures] = useState<string[]>(Array(20).fill(''));
  const { state, result, errorMsg, submit } = usePrediction();

  const handleGenerateRandom = () => {
    const randomFeatures = Array.from({ length: 20 }, () => (Math.random() * 10 - 5).toFixed(4));
    setFeatures(randomFeatures);
  };

  const handleFeatureChange = (index: number, value: string) => {
    const newFeatures = [...features];
    newFeatures[index] = value;
    setFeatures(newFeatures);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const parsedFeatures = features.map(f => parseFloat(f));
    if (parsedFeatures.some(isNaN)) {
      alert('Please ensure all features are valid numbers.');
      return;
    }
    submit(parsedFeatures);
  };

  const chartData = (result?.features || []).map((val, idx) => ({
    name: `F${idx}`,
    value: val,
  }));

  const isAnomaly = result?.prediction === 'Anomaly';

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-900 p-8 font-sans">
      <div className="max-w-6xl mx-auto space-y-8">
        <div>
          <h1 className="text-4xl font-bold tracking-tight text-slate-900 dark:text-white">Quantum SVM Anomaly Detection</h1>
          <p className="text-lg text-slate-500 dark:text-slate-400 mt-2">Financial Time-Series Analysis Dashboard</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* ── Input Form Column ── */}
          <div className="lg:col-span-1 space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Input Features</CardTitle>
                <CardDescription>Enter exactly 20 numeric features for prediction.</CardDescription>
              </CardHeader>
              <CardContent>
                <form id="prediction-form" onSubmit={handleSubmit} className="space-y-4">
                  <Button type="button" variant="outline" className="w-full flex items-center gap-2" onClick={handleGenerateRandom}>
                    <Wand2 className="w-4 h-4" /> Random Test Data
                  </Button>

                  <div className="grid grid-cols-2 gap-3 max-h-[500px] overflow-y-auto pr-2 rounded-md border p-4 bg-slate-50 dark:bg-slate-950/50">
                    {features.map((val, idx) => (
                      <div key={idx} className="space-y-1">
                        <label className="text-xs font-medium text-slate-500">Feature {idx}</label>
                        <Input
                          type="number"
                          step="any"
                          required
                          placeholder="0.0"
                          value={val}
                          onChange={(e) => handleFeatureChange(idx, e.target.value)}
                          className="text-sm h-8"
                        />
                      </div>
                    ))}
                  </div>
                </form>
              </CardContent>
              <CardFooter>
                <Button
                  type="submit"
                  form="prediction-form"
                  className="w-full"
                  disabled={state === 'submitting' || state === 'polling_backend'}
                >
                  {(state === 'submitting' || state === 'polling_backend') ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      {state === 'submitting' ? 'Submitting...' : 'Analyzing...'}
                    </>
                  ) : 'Run Anomaly Detection'}
                </Button>
              </CardFooter>
            </Card>
          </div>

          {/* ── Status & Results Column ── */}
          <div className="lg:col-span-2 space-y-6">

            {/* Processing indicator */}
            {(state === 'submitting' || state === 'polling_backend') && (
              <Card className="border-blue-200 bg-blue-50 dark:bg-blue-950/20 dark:border-blue-800">
                <CardHeader className="pb-4">
                  <CardTitle className="text-blue-700 dark:text-blue-300 flex items-center gap-2">
                    <Loader2 className="w-5 h-5 animate-spin" />
                    Processing Prediction
                  </CardTitle>
                  <CardDescription className="text-blue-600/80 dark:text-blue-400/80">
                    {state === 'submitting' ? 'Sending data to backend...' : 'Running Quantum SVM... Polling backend status.'}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Progress value={state === 'submitting' ? 25 : 75} className="h-2" />
                </CardContent>
              </Card>
            )}

            {/* Error alert */}
            {state === 'error' && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{errorMsg || 'An unknown error occurred.'}</AlertDescription>
              </Alert>
            )}

            {/* ── Success: Results ── */}
            {state === 'success' && result && (
              <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">

                {/* Prediction Label Card */}
                <Card className={isAnomaly
                  ? 'border-red-500 bg-red-50 dark:bg-red-950/20'
                  : 'border-emerald-500 bg-emerald-50 dark:bg-emerald-950/20'
                }>
                  <CardHeader>
                    <CardTitle className={`flex items-center gap-3 text-2xl ${isAnomaly ? 'text-red-600 dark:text-red-400' : 'text-emerald-600 dark:text-emerald-400'}`}>
                      {isAnomaly
                        ? <ShieldAlert className="w-7 h-7" />
                        : <ShieldCheck className="w-7 h-7" />}
                      {isAnomaly ? 'Anomaly Detected' : 'Normal Activity'}
                    </CardTitle>
                  </CardHeader>
                </Card>

                {/* Confidence Gauge Card */}
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">Confidence Gauge</CardTitle>
                    <CardDescription>
                      Normalized from the SVM decision-function distance via sigmoid mapping.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex items-end justify-between">
                      <span className={`text-4xl font-bold font-mono ${isAnomaly ? 'text-red-500' : 'text-emerald-500'}`}>
                        {result.confidence_pct.toFixed(1)}%
                      </span>
                      <span className="text-sm text-slate-500 font-mono">
                        raw score: {result.confidence_score.toFixed(4)}
                      </span>
                    </div>
                    {/* Gauge bar */}
                    <div className="w-full h-4 rounded-full bg-slate-200 dark:bg-slate-800 overflow-hidden">
                      <div
                        className={`h-full rounded-full transition-all duration-700 ease-out ${isAnomaly ? 'bg-red-500' : 'bg-emerald-500'}`}
                        style={{ width: `${Math.min(result.confidence_pct, 100)}%` }}
                      />
                    </div>
                    <div className="flex justify-between text-xs text-slate-400">
                      <span>0 % — uncertain</span>
                      <span>100 % — high confidence</span>
                    </div>
                  </CardContent>
                </Card>

                {/* Feature Vector Chart */}
                <Card>
                  <CardHeader>
                    <CardTitle>Analyzed Features</CardTitle>
                    <CardDescription>Bar chart representation of the 20 input features vector.</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="h-[300px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                          <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                          <XAxis dataKey="name" axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#64748b' }} dy={10} />
                          <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 12, fill: '#64748b' }} />
                          <Tooltip
                            cursor={{ fill: 'rgba(226, 232, 240, 0.4)' }}
                            contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)' }}
                          />
                          <Bar dataKey="value" fill={isAnomaly ? '#ef4444' : '#10b981'} radius={[4, 4, 0, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Idle placeholder */}
            {state === 'idle' && (
              <Card className="border-dashed border-2 flex items-center justify-center h-full min-h-[400px] bg-slate-50/50 dark:bg-slate-900/50">
                <div className="text-center text-slate-500">
                  <Wand2 className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Awaiting data input.</p>
                  <p className="text-sm">Enter 20 features and run the anomaly detection.</p>
                </div>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
