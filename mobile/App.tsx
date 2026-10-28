import React, { useEffect, useState } from 'react';
import { Button, SafeAreaView, StyleSheet, Text, View } from 'react-native';
import { requireNativeModule, requireNativeViewManager } from 'expo-modules-core';
import RimTest from './RimTest';

type Status = { message: string; tracking?: string; frames?: number; recording?: boolean; path?: string; region?: boolean; surfaces?: number; walls?: number; points?: number; saving?: boolean; overlays?: number; overlaysInView?: number; build?: string; depthMode?: 'lidar' | 'multiview'; lidarSupported?: boolean; cloudPoints?: number; tracks?: number };
type CaptureProps = { style: object; recording: boolean; regionSize: number; captureWithoutBox: boolean; resetToken: number; forceMultiView: boolean; onStatus: (event: { nativeEvent: Status }) => void };
const CaptureView = requireNativeViewManager<CaptureProps>('TareCapture');
const capture = requireNativeModule<{ shareRecording(path: string): Promise<void>; latestRecording(): Promise<string | null> }>('TareCapture');

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
  const [rimMode, setRimMode] = useState(false);
  useEffect(() => {
    capture.latestRecording().then(saved => { if (saved) setPath(saved); })
      .catch(error => setStatus({ message: `Could not load saved recordings: ${error}` }));
  }, []);
  return <SafeAreaView style={styles.page}>
    <Text style={styles.title}>Tare Capture{status.build ? ` · Build ${status.build}` : ''}</Text>
    <Text style={styles.copy}>Look downward and move slowly sideways over a textured table. Tap a cyan surface near the object, or skip the box.</Text>
    <CaptureView style={styles.camera} recording={recording} regionSize={size} captureWithoutBox={captureWithoutBox} resetToken={resetToken} forceMultiView={forceMultiView} onStatus={({ nativeEvent: e }) => {
      setStatus(e);
      if (e.region !== undefined) setRegion(e.region);
      if (e.recording !== undefined) setRecording(e.recording);
      if (e.path) setPath(e.path);
    }} />
    <Text style={styles.copy}>{status.message}</Text>
    <Button title={rimMode ? 'Back to scanning' : 'Plate rim test'} disabled={recording || status.saving} onPress={() => setRimMode(!rimMode)} />
    {rimMode ? <RimTest disabled={status.tracking !== 'normal'} /> : <>
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
    </>}
  </SafeAreaView>;
}
const styles = StyleSheet.create({
  page: { flex: 1, backgroundColor: '#f5f5f5', padding: 16 },
  title: { fontSize: 24, fontWeight: '600' },
  copy: { paddingVertical: 6, color: '#333' },
  camera: { flex: 1, minHeight: 180, borderRadius: 12, overflow: 'hidden' },
  row: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
});
