import React, { useEffect, useState } from 'react';
import { Button, SafeAreaView, StyleSheet, Text, View } from 'react-native';
import { requireNativeModule, requireNativeViewManager } from 'expo-modules-core';

type Status = { message: string; tracking?: string; frames?: number; recording?: boolean; path?: string; region?: boolean };
type CaptureProps = { style: object; recording: boolean; regionSize: number; onStatus: (event: { nativeEvent: Status }) => void };
const CaptureView = requireNativeViewManager<CaptureProps>('TareCapture');
const capture = requireNativeModule<{ shareRecording(path: string): Promise<void>; latestRecording(): Promise<string | null> }>('TareCapture');

export default function App() {
  const [recording, setRecording] = useState(false);
  const [status, setStatus] = useState<Status>({ message: 'Allow camera access, then tap the object to place a scan box.' });
  const [path, setPath] = useState<string>();
  const [region, setRegion] = useState(false);
  const [size, setSize] = useState(1);
  const [sharing, setSharing] = useState(false);
  useEffect(() => {
    capture.latestRecording().then(saved => { if (saved) setPath(saved); })
      .catch(error => setStatus({ message: `Could not load saved recordings: ${error}` }));
  }, []);
  return <SafeAreaView style={styles.page}>
    <Text style={styles.title}>Tare Capture</Text>
    <Text style={styles.copy}>Tap the object’s support surface to place a box. Keep the object stationary and orbit slowly.</Text>
    <CaptureView style={styles.camera} recording={recording} regionSize={size} onStatus={({ nativeEvent: e }) => {
      setStatus(e);
      if (e.region !== undefined) setRegion(e.region);
      if (e.recording !== undefined) setRecording(e.recording);
      if (e.path) setPath(e.path);
    }} />
    <Text style={styles.copy}>{status.message}</Text>
    <Text style={styles.copy}>Tracking: {status.tracking || 'initializing'} · Frames: {status.frames || 0}</Text>
    <View style={styles.row}>
      <Button title="Smaller box" disabled={recording} onPress={() => setSize(Math.max(0.25, size - 0.25))} />
      <Text>{size.toFixed(2)} m</Text>
      <Button title="Larger box" disabled={recording} onPress={() => setSize(Math.min(3, size + 0.25))} />
    </View>
    <Button title={recording ? 'Stop and save' : 'Start scan'} disabled={!region && !recording} onPress={() => setRecording(!recording)} />
    <Button title={sharing ? 'Preparing export…' : 'Export last recording'} disabled={!path || recording || sharing} onPress={async () => {
      if (!path) return;
      setSharing(true);
      try { await capture.shareRecording(path); }
      catch (error) { setStatus({ message: String(error) }); }
      finally { setSharing(false); }
    }} />
    <Text style={styles.copy}>Maximum 120 seconds / 240 frames. RGB and poses only; no measured depth. Exports stay on this phone until you share them.</Text>
  </SafeAreaView>;
}
const styles = StyleSheet.create({
  page: { flex: 1, backgroundColor: '#f5f5f5', padding: 16 },
  title: { fontSize: 24, fontWeight: '600' },
  copy: { paddingVertical: 6, color: '#333' },
  camera: { flex: 1, minHeight: 180, borderRadius: 12, overflow: 'hidden' },
  row: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
});
