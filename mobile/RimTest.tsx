import React, { useState } from 'react';
import { Button, Image, Modal, Pressable, SafeAreaView, ScrollView, StyleSheet, Text, TextInput, View, useWindowDimensions } from 'react-native';
import { requireNativeModule } from 'expo-modules-core';

// Temporary "tap the rim" test: stands in for the rim detector so the two-view rim
// triangulation can run on real plates. Taps are stored in portrait image pixels
// (pixel centers at integer coordinates), which the native side maps to sensor pixels.

type Point = [number, number];
type Shot = { uri: string; width: number; height: number; tiltDeg: number; cameraHeightM: number; hasDepth: boolean; baselineM?: number };
type Rim = { diameterMm: number; heightMm: number; rmsPx: number; confidence: number; diameterSigmaMm?: number; heightSigmaMm?: number };
type Result = {
  twoView?: Rim; twoViewError?: string; baselineM?: number; cameraHeightsM: number[];
  singleView: { slot: number; onTable?: Rim; atMeasuredHeight?: Rim }[];
  lidar?: { diameterMm: number; heightMm: number; boundaryRmsMm: number; twoViewDiameterErrorMm?: number; twoViewDiameterErrorPct?: number; twoViewHeightErrorMm?: number };
  rulerErrorMm?: number; rulerErrorPct?: number; savedTo?: string;
};
const rim = requireNativeModule<{
  rimSnapshot(slot: number): Promise<Shot>;
  rimMeasure(taps: Point[][], rulerDiameterMm: number | null): Promise<Result>;
  rimReset(): Promise<void>;
}>('TareCapture');

const MIN_TAPS = 6;
const NAMES = ['Overhead', 'Side'];
const mm = (value: number, sigma?: number) => `${value.toFixed(1)}${sigma !== undefined ? ` ± ${sigma.toFixed(1)}` : ''} mm`;

export default function RimTest({ disabled }: { disabled: boolean }) {
  const [shots, setShots] = useState<(Shot | undefined)[]>([undefined, undefined]);
  const [taps, setTaps] = useState<Point[][]>([[], []]);
  const [editing, setEditing] = useState<number>();
  const [ruler, setRuler] = useState('');
  const [result, setResult] = useState<Result>();
  const [message, setMessage] = useState<string>();
  const [busy, setBusy] = useState(false);

  async function capture(slot: number) {
    setBusy(true); setMessage(undefined); setResult(undefined);
    try {
      const shot = await rim.rimSnapshot(slot);
      setShots(current => current.map((s, i) => (i === slot ? shot : s)));
      setTaps(current => current.map((t, i) => (i === slot ? [] : t)));
      setEditing(slot);
    } catch (error) { setMessage(String(error)); }
    finally { setBusy(false); }
  }
  async function measure() {
    setBusy(true); setMessage(undefined);
    try {
      const value = parseFloat(ruler);
      setResult(await rim.rimMeasure(taps, Number.isFinite(value) ? value : null));
    } catch (error) { setMessage(String(error)); }
    finally { setBusy(false); }
  }
  async function reset() {
    await rim.rimReset();
    setShots([undefined, undefined]); setTaps([[], []]); setResult(undefined); setMessage(undefined);
  }

  const ready = shots.every(Boolean) && taps.every(t => t.length >= MIN_TAPS);
  return <View>
    <Text style={styles.copy}>1. Hold the phone above the plate looking down and capture Overhead. 2. Step 15–30 cm to the side, lower the phone so the plate is seen at an angle, and capture Side. 3. Tap {MIN_TAPS}+ points spread around the rim's outer edge in each photo (pinch to zoom).</Text>
    {[0, 1].map(slot => {
      const shot = shots[slot];
      return <View key={slot} style={styles.row}>
        <Button title={`${shot ? 'Retake' : 'Capture'} ${NAMES[slot]}`} disabled={disabled || busy} onPress={() => capture(slot)} />
        {shot ? <Button title={`Tap rim (${taps[slot].length})`} onPress={() => setEditing(slot)} /> : null}
        <Text style={styles.small}>{shot ? `tilt ${shot.tiltDeg.toFixed(0)}° · ${(shot.cameraHeightM * 100).toFixed(0)} cm up${shot.baselineM !== undefined ? ` · ${(shot.baselineM * 100).toFixed(0)} cm apart` : ''}` : ''}</Text>
      </View>;
    })}
    <View style={styles.row}>
      <Text>True diameter (optional):</Text>
      <TextInput style={styles.input} value={ruler} onChangeText={setRuler} placeholder="mm" keyboardType="decimal-pad" />
    </View>
    <View style={styles.row}>
      <Button title={busy ? 'Working…' : 'Measure rim'} disabled={!ready || busy} onPress={measure} />
      <Button title="Reset rim test" disabled={busy} onPress={reset} />
    </View>
    {message ? <Text style={styles.error}>{message}</Text> : null}
    {result ? <ResultView result={result} /> : null}
    {editing !== undefined && shots[editing] ? <TapEditor
      shot={shots[editing]!} title={NAMES[editing]} points={taps[editing]}
      onChange={points => setTaps(current => current.map((t, i) => (i === editing ? points : t)))}
      onDone={() => setEditing(undefined)} /> : null}
  </View>;
}

function ResultView({ result }: { result: Result }) {
  const lines: string[] = [];
  if (result.twoView) {
    const r = result.twoView;
    lines.push(`Two-view: Ø ${mm(r.diameterMm, r.diameterSigmaMm)} · rim height ${mm(r.heightMm, r.heightSigmaMm)}`);
    lines.push(`Fit ${r.rmsPx.toFixed(1)} px RMS · confidence ${r.confidence.toFixed(2)}${result.baselineM !== undefined ? ` · baseline ${(result.baselineM * 100).toFixed(0)} cm` : ''}`);
  } else lines.push(`Two-view failed: ${result.twoViewError}`);
  for (const single of result.singleView) {
    const parts = [single.onTable && `rim on table Ø ${mm(single.onTable.diameterMm)}`,
      single.atMeasuredHeight && `at measured height Ø ${mm(single.atMeasuredHeight.diameterMm)}`].filter(Boolean);
    if (parts.length) lines.push(`${NAMES[single.slot]} only: ${parts.join(' · ')}`);
  }
  if (result.lidar) {
    const l = result.lidar;
    lines.push(`LiDAR reference: Ø ${mm(l.diameterMm)} · rim height ${mm(l.heightMm)}`);
    if (l.twoViewDiameterErrorMm !== undefined) {
      lines.push(`Two-view vs LiDAR: Ø ${l.twoViewDiameterErrorMm >= 0 ? '+' : ''}${l.twoViewDiameterErrorMm.toFixed(1)} mm (${l.twoViewDiameterErrorPct!.toFixed(1)}%) · height ${l.twoViewHeightErrorMm!.toFixed(1)} mm`);
    }
  }
  if (result.rulerErrorMm !== undefined) {
    lines.push(`Two-view vs ruler: ${result.rulerErrorMm >= 0 ? '+' : ''}${result.rulerErrorMm.toFixed(1)} mm (${result.rulerErrorPct!.toFixed(1)}%)`);
  }
  if (result.savedTo) lines.push(`Saved to ${result.savedTo.split('/Documents/')[1] ?? result.savedTo}`);
  return <View style={styles.result}>{lines.map((line, i) => <Text key={i} style={i === 0 ? styles.headline : styles.small}>{line}</Text>)}</View>;
}

function TapEditor({ shot, title, points, onChange, onDone }:
  { shot: Shot; title: string; points: Point[]; onChange: (points: Point[]) => void; onDone: () => void }) {
  const { width } = useWindowDimensions();
  const [zoom, setZoom] = useState(1);
  const displayHeight = width * shot.height / shot.width;
  const scale = shot.width / width; // Image pixels per point.
  const dot = 8 / zoom;
  return <Modal animationType="slide" onRequestClose={onDone}>
    <SafeAreaView style={styles.editor}>
      <Text style={styles.editorTitle}>{title}: tap the rim's outer edge ({points.length}/{MIN_TAPS}+)</Text>
      <ScrollView style={{ flex: 1 }} maximumZoomScale={8} minimumZoomScale={1} bouncesZoom={false} scrollEventThrottle={16}
        onScroll={event => setZoom((event.nativeEvent as { zoomScale?: number }).zoomScale ?? 1)}
        centerContent>
        <Pressable onPress={event => {
          const { locationX, locationY } = event.nativeEvent;
          onChange([...points, [locationX * scale - 0.5, locationY * scale - 0.5]]);
        }}>
          <Image source={{ uri: shot.uri }} style={{ width, height: displayHeight }} />
          {points.map(([x, y], i) => <View key={i} pointerEvents="none" style={{
            position: 'absolute', left: (x + 0.5) / scale - dot / 2, top: (y + 0.5) / scale - dot / 2,
            width: dot, height: dot, borderRadius: dot / 2, borderWidth: 1.5 / zoom, borderColor: '#ff2d55',
          }} />)}
        </Pressable>
      </ScrollView>
      <View style={styles.row}>
        <Button title="Undo" disabled={!points.length} onPress={() => onChange(points.slice(0, -1))} />
        <Button title="Clear" disabled={!points.length} onPress={() => onChange([])} />
        <Button title="Done" onPress={onDone} />
      </View>
    </SafeAreaView>
  </Modal>;
}

const styles = StyleSheet.create({
  copy: { paddingVertical: 6, color: '#333' },
  small: { color: '#333', flexShrink: 1 },
  row: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 8, paddingVertical: 2 },
  input: { borderWidth: 1, borderColor: '#bbb', borderRadius: 6, paddingHorizontal: 8, paddingVertical: 4, minWidth: 80, backgroundColor: '#fff' },
  error: { color: '#b00020', paddingVertical: 6 },
  result: { paddingVertical: 6, gap: 2 },
  headline: { fontWeight: '600', color: '#111' },
  editor: { flex: 1, backgroundColor: '#000' },
  editorTitle: { color: '#fff', padding: 12 },
});
