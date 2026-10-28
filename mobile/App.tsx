import React, { useEffect, useState } from 'react';
import { Button, SafeAreaView, StyleSheet, Text, View } from 'react-native';
import { requireNativeModule, requireNativeViewManager } from 'expo-modules-core';
import RimTest from './RimTest';

type Demo = { stage: 'idle' | 'capturing' | 'done'; views: number; taps: number; coverage: number; fitting: boolean; spots: number; hint?: string; fitError?: string; rim?: { diameterMm: number; heightMm: number; support: number }; lidar?: { diameterMm: number; heightMm: number } };
type Status = { demo?: Demo; message: string; tracking?: string; frames?: number; recording?: boolean; path?: string; region?: boolean; surfaces?: number; walls?: number; points?: number; saving?: boolean; overlays?: number; overlaysInView?: number; build?: string; depthMode?: 'lidar' | 'multiview'; lidarSupported?: boolean; cloudPoints?: number; tracks?: number };
type CaptureProps = { style: object; recording: boolean; regionSize: number; captureWithoutBox: boolean; resetToken: number; forceMultiView: boolean; demoMode: boolean; onStatus: (event: { nativeEvent: Status }) => void };
const CaptureView = requireNativeViewManager<CaptureProps>('TareCapture');
const capture = requireNativeModule<{ shareRecording(path: string): Promise<void>; latestRecording(): Promise<string | null>; demoFinish(): Promise<void>; demoReset(): Promise<void> }>('TareCapture');
type Mode = 'demo' | 'rim' | 'scan';
const MODES: [Mode, string][] = [['demo', 'Plate demo'], ['rim', 'Rim taps'], ['scan', 'Scan']];

export default function App() {
  const [recording, setRecording] = useState(false);
  const [status, setStatus] = useState<Status>({ message: 'Allow camera access, then tap the object to place a scan box.' });
  const [path, setPath] = useState<string>();
  const [region, setRegion] = useState(false);
  const [size, setSize] = useState(1);
  const [sharing, setSharing] = useState(false);
  const [captureWithoutBox, setCaptureWithoutBox] = useState(false);
  const [resetToken, setResetToken] = useState(0);
  const [forceMultiView, setForceMultiView] = useState(false);
  const [mode, setMode] = useState<Mode>('demo');
  useEffect(() => {
    capture.latestRecording().then(saved => { if (saved) setPath(saved); })
      .catch(error => setStatus({ message: `Could not load saved recordings: ${error}` }));
  }, []);
  return <SafeAreaView style={styles.page}>
    <Text style={styles.title}>Tare Capture{status.build ? ` · Build ${status.build}` : ''}</Text>
    <View style={styles.row}>
      {MODES.map(([value, label]) => <Button key={value} title={mode === value ? `• ${label}` : label} disabled={recording || status.saving} onPress={() => setMode(value)} />)}
    </View>
    <CaptureView style={styles.camera} recording={recording} regionSize={size} captureWithoutBox={captureWithoutBox} resetToken={resetToken} forceMultiView={forceMultiView} demoMode={mode === 'demo'} onStatus={({ nativeEvent: e }) => {
      setStatus(e);
      if (e.region !== undefined) setRegion(e.region);
      if (e.recording !== undefined) setRecording(e.recording);
      if (e.path) setPath(e.path);
    }} />
    <Text style={mode === 'demo' ? styles.headline : styles.copy}>{status.message}</Text>
    {mode === 'demo' ? <DemoPanel demo={status.demo} lidar={status.depthMode === 'lidar'} /> : null}
    {mode === 'rim' ? <RimTest disabled={status.tracking !== 'normal'} /> : null}
    {mode === 'scan' ? <>
    <Text style={styles.copy}>Tracking: {status.tracking || 'initializing'} · Surfaces: {status.surfaces || 0} · Frames: {status.frames || 0}</Text>
    <Text style={styles.copy}>Walls: {status.walls || 0} · Feature points: {status.points || 0}</Text>
    <Text style={styles.copy}>Surface overlays: {status.overlays || 0} · In view: {status.overlaysInView || 0}</Text>
    <Text style={styles.copy}>Depth: {status.depthMode === 'lidar' ? 'LiDAR' : 'multi-view (OpenCV)'} · Cloud points: {status.cloudPoints || 0}{status.depthMode === 'multiview' ? ` · Tracks: ${status.tracks || 0}` : ''}</Text>
    {status.lidarSupported ? <Button title={forceMultiView ? 'Use LiDAR depth' : 'Force multi-view depth (benchmark)'} disabled={recording || status.saving} onPress={() => setForceMultiView(!forceMultiView)} /> : null}
    <Button title="Restart surface detection" disabled={recording || status.saving} onPress={() => setResetToken(value => value + 1)} />
    <View style={styles.row}>
      <Button title="Smaller box" disabled={recording} onPress={() => setSize(Math.max(0.25, size - 0.25))} />
      <Text>{size.toFixed(2)} m</Text>
      <Button title="Larger box" disabled={recording} onPress={() => setSize(Math.min(3, size + 0.25))} />
    </View>
    <Button title={captureWithoutBox ? 'Require scan box' : 'Skip box — capture images and poses'} disabled={recording} onPress={() => setCaptureWithoutBox(!captureWithoutBox)} />
    <Button title={recording ? 'Stop and save' : 'Start scan'} disabled={!region && !captureWithoutBox && !recording} onPress={() => setRecording(!recording)} />
    <Button title={sharing ? 'Preparing export…' : 'Export last recording'} disabled={!path || recording || sharing} onPress={async () => {
      if (!path) return;
      setSharing(true);
      try { await capture.shareRecording(path); }
      catch (error) { setStatus({ message: String(error) }); }
      finally { setSharing(false); }
    }} />
    <Text style={styles.copy}>Maximum 120 seconds / 240 frames. RGB, poses and a fused point cloud (points.ply). Exports stay on this phone until you share them.</Text>
    </> : null}
  </SafeAreaView>;
}

function DemoPanel({ demo, lidar }: { demo?: Demo; lidar: boolean }) {
  if (!demo) return null;
  const rim = demo.rim;
  return <View>
    {demo.stage !== 'idle' ? <Text style={styles.copy}>
      {demo.views} views · {Math.round(demo.coverage * 100)}% of the hemisphere{demo.taps > 1 ? ` · ${demo.taps} taps` : ''}{demo.fitting ? ' · fitting…' : ''}
    </Text> : null}
    {rim ? <Text style={styles.copy}>Rim Ø {rim.diameterMm.toFixed(0)} mm · {rim.heightMm.toFixed(0)} mm high · {demo.spots} probe spots 1 in inside</Text> : null}
    {demo.lidar && rim ? <Text style={styles.copy}>LiDAR check: Ø {demo.lidar.diameterMm.toFixed(0)} mm ({(rim.diameterMm - demo.lidar.diameterMm >= 0 ? '+' : '')}{(rim.diameterMm - demo.lidar.diameterMm).toFixed(0)} mm) · {demo.lidar.heightMm.toFixed(0)} mm high</Text>
      : demo.stage === 'done' && lidar && rim ? <Text style={styles.copy}>LiDAR check unavailable for this plate.</Text> : null}
    {demo.stage === 'capturing' ? <Text style={styles.small}>Tap the plate's middle again any time to help it lock on.</Text> : null}
    <View style={styles.row}>
      <Button title="Finish now" disabled={demo.stage !== 'capturing' || demo.views < 6} onPress={() => capture.demoFinish()} />
      <Button title="Restart" disabled={demo.stage === 'idle'} onPress={() => capture.demoReset()} />
    </View>
  </View>;
}
const styles = StyleSheet.create({
  page: { flex: 1, backgroundColor: '#f5f5f5', padding: 16 },
  title: { fontSize: 24, fontWeight: '600' },
  copy: { paddingVertical: 6, color: '#333' },
  small: { paddingVertical: 4, color: '#666' },
  headline: { paddingVertical: 8, fontSize: 17, fontWeight: '600', color: '#111' },
  camera: { flex: 1, minHeight: 180, borderRadius: 12, overflow: 'hidden' },
  row: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
});
