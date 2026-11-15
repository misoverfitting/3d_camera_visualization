import React, { useEffect, useState } from 'react';
import { Alert, ImageBackground, Pressable, SafeAreaView, ScrollView, StyleSheet, Text, View } from 'react-native';
import { requireNativeModule, requireNativeViewManager } from 'expo-modules-core';
import RimTest from './RimTest';
import { useFonts } from 'expo-font';
import { StatusBar } from 'expo-status-bar';
import { brand, Button, TareLogo } from './Brand';

type Demo = { stage: 'idle' | 'capturing' | 'done'; views: number; taps: number; coverage: number; tableSet?: boolean; tableOffsetMm?: number; setupStep?: 'table' | 'plate' | 'capture'; coveredCellsByBand?: number[]; coverageCells?: boolean[][]; imageBuildFailures?: number; fitReady?: boolean; rejections?: Record<string, number>; fitting: boolean; spots: number; hint?: string; fitError?: string; rim?: { diameterMm: number; heightMm: number; support: number }; lidar?: { diameterMm: number; heightMm: number }; heightOffsetMm?: number; hasPhoto?: boolean; food?: Food; probe?: { running: boolean; done: number; total: number } };
type FoodSource = 'on-device' | 'claude';
type Food = { phase: 'idle' | 'working' | 'done' | 'failed'; error?: string; source?: FoodSource; model?: string; dish?: { label: string; confidence: number; alternatives: string[] }; saved?: string; savedCount?: number; items: { label: string; category: string; confidence: number; color: string; points: number; alternatives?: string[] }[] };
type Status = { demo?: Demo; message: string; tracking?: string; frames?: number; recording?: boolean; path?: string; region?: boolean; surfaces?: number; walls?: number; points?: number; saving?: boolean; overlays?: number; overlaysInView?: number; build?: string; depthMode?: 'lidar' | 'multiview'; lidarSupported?: boolean; cloudPoints?: number; tracks?: number };
type CaptureProps = { style: object; recording: boolean; regionSize: number; captureWithoutBox: boolean; resetToken: number; forceMultiView: boolean; demoMode: boolean; onStatus: (event: { nativeEvent: Status }) => void };
const CaptureView = requireNativeViewManager<CaptureProps>('TareCapture');
const capture = requireNativeModule<{ shareRecording(path: string): Promise<void>; latestRecording(): Promise<string | null>; demoFinish(): Promise<void>; demoReset(): Promise<void>; demoNudge(deltaMm: number): Promise<void>; demoAdjustTable(deltaMm: number): Promise<void>; demoIdentify(source: FoodSource): Promise<Food>; demoProbe(): Promise<void>; shareFoodScans(): Promise<void> }>('TareCapture');
type Mode = 'demo' | 'rim' | 'scan';
const MODES: [Mode, string][] = [['demo', 'Plate scan'], ['rim', 'Rim test'], ['scan', '3D capture']];

export default function App() {
  const [fontsLoaded, fontError] = useFonts({
    InstrumentSans: require('./assets/brand/InstrumentSans.ttf'),
    SometypeMono: require('./assets/brand/SometypeMono.ttf'),
  });
  const [recording, setRecording] = useState(false);
  const [status, setStatus] = useState<Status>({ message: 'Allow camera access, then point at bare table beside the plate.' });
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
  if (!fontsLoaded && !fontError) return <SafeAreaView style={styles.page}><View style={styles.loading}><TareLogo /></View></SafeAreaView>;
  return <SafeAreaView style={styles.page}>
    <StatusBar style="dark" />
    <View style={styles.topLine} />
    <ImageBackground source={require('./assets/brand/plate-horizon-glow-2.jpg')} imageStyle={{ opacity: 0.18 }} style={styles.header}>
      <View style={styles.headerRow}>
        <View style={styles.logoBlock}>
          <TareLogo width={100} />
          <Text style={styles.buildLabel}>{status.build ? `BUILD ${status.build}` : 'CAPTURE / 03'}</Text>
        </View>
        <View style={styles.tabs} accessibilityRole="tablist">
          {MODES.map(([value, label]) => <Pressable key={value} accessibilityRole="tab" accessibilityLabel={label} accessibilityState={{ selected: mode === value, disabled: recording || !!status.saving }}
            disabled={recording || status.saving} onPress={() => setMode(value)} style={[styles.tab, mode === value && styles.activeTab, (recording || status.saving) && { opacity: 0.4 }]}>
            <Text numberOfLines={2} maxFontSizeMultiplier={1.4} style={[styles.tabLabel, mode === value && styles.activeTabLabel]}>{label}</Text>
          </Pressable>)}
        </View>
      </View>
    </ImageBackground>
    <View style={styles.cameraFrame}>
    <CaptureView style={styles.camera} recording={recording} regionSize={size} captureWithoutBox={captureWithoutBox} resetToken={resetToken} forceMultiView={forceMultiView} demoMode={mode === 'demo'} onStatus={({ nativeEvent: e }) => {
      setStatus(e);
      if (e.region !== undefined) setRegion(e.region);
      if (e.recording !== undefined) setRecording(e.recording);
      if (e.path) setPath(e.path);
    }} />
    <View pointerEvents="none" style={styles.cameraTag}><View style={styles.liveDot} /><Text style={styles.cameraTagText}>LIVE / {status.tracking === 'normal' ? 'READY' : 'FINDING SURFACE'}</Text></View>
    </View>
    <ScrollView style={styles.controls} contentContainerStyle={styles.controlsContent} keyboardShouldPersistTaps="handled">
    <Text style={styles.eyebrow}>{mode === 'demo' ? (status.demo?.setupStep === 'table' ? '01 / SET THE TABLE SURFACE' : status.demo?.setupStep === 'plate' ? '02 / SELECT THE PLATE CENTER' : '03 / CAPTURE YOUR PLATE') : mode === 'rim' ? '02 / CHECK THE RIM' : '03 / RECORD A CAPTURE'}</Text>
    <Text accessibilityLiveRegion="polite" style={styles.headline}>{status.message}</Text>
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
      <Text style={styles.copy}>{size.toFixed(2)} m</Text>
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
    </ScrollView>
    {mode === 'demo' ? <View style={styles.actionDock}>
      <View style={{ flex: 1 }}><Button primary title="Finish scan" disabled={status.demo?.stage !== 'capturing' || (status.demo?.views ?? 0) < 6} onPress={() => capture.demoFinish()} /></View>
      <View style={{ flex: 1 }}><Button title="Restart" disabled={!status.demo || (status.demo.stage === 'idle' && !status.demo.tableSet)} onPress={() => capture.demoReset()} /></View>
    </View> : null}
  </SafeAreaView>;
}

// Errors arrive in the status as food.error; the ring spots are probed instead.
const identifyThenProbe = (source: FoodSource) => { capture.demoIdentify(source).catch(() => {}).then(() => capture.demoProbe()); };

function DemoPanel({ demo, lidar }: { demo?: Demo; lidar: boolean }) {
  const [approved, setApproved] = useState(false);
  const [details, setDetails] = useState(false);
  const source: FoodSource = 'on-device';
  const fitted = demo?.stage === 'done' && !!demo.rim && !demo.fitting && !!demo.fitReady;
  const ready = fitted && !!demo?.hasPhoto;
  useEffect(() => { if (demo?.stage !== 'done') setApproved(false); }, [demo?.stage]);
  if (!demo) return null;
  const rim = demo.rim;
  const food = demo.food;
  return <View>
    {demo.stage === 'idle' && demo.tableSet ? <View style={styles.row}>
      <Button title="Surface down 5 mm" onPress={() => capture.demoAdjustTable(-5)} />
      <Button title="Surface up 5 mm" onPress={() => capture.demoAdjustTable(5)} />
    </View> : null}
    {demo.stage !== 'idle' ? <View style={styles.coverageCard}>
      <View style={styles.row}><Text style={styles.eyebrow}>COVERAGE</Text><Text style={styles.coverageNumber}>{Math.round(demo.coverage * 100)}%</Text></View>
      <View style={styles.track}><View style={[styles.fill, { width: `${Math.min(100, Math.max(0, demo.coverage * 100))}%` }]} /></View>
      {[1, 0].map(band => <View key={band} style={styles.cellRow} accessibilityLabel={`${band === 1 ? 'High' : 'Lower'} coverage: ${demo.coveredCellsByBand?.[band] ?? 0} of 12 viewpoints`}><Text style={styles.cellLabel}>{band === 1 ? 'HIGH' : 'LOWER'}</Text>{Array.from({ length: 12 }, (_, cell) => <View key={cell} style={[styles.cell, demo.coverageCells?.[band]?.[cell] && styles.cellCovered]} />)}</View>)}
      <View style={styles.row}><Text style={styles.small}>HIGH VIEWS  {demo.coveredCellsByBand?.[1] ?? 0}/12</Text><Text style={styles.small}>LOWER VIEWS  {demo.coveredCellsByBand?.[0] ?? 0}/12</Text></View>
    </View> : null}
    {ready && !approved ? <Button primary title="Use this plate & identify food" onPress={() => { setApproved(true); identifyThenProbe(source); }} /> : null}
    <Pressable accessibilityRole="button" accessibilityState={{ expanded: details }} onPress={() => setDetails(!details)} style={{ paddingVertical: 12 }}><Text style={styles.small}>{details ? 'HIDE SCAN DETAILS' : 'SHOW SCAN DETAILS'}</Text></Pressable>
    {details ? <View>
    {demo.tableSet ? <Text style={styles.small}>Surface adjustment: {Math.round(demo.tableOffsetMm ?? 0)} mm</Text> : null}
    {!lidar && demo.tableSet ? <Text style={styles.small}>Point heights are estimated from the table and plate geometry, not measured food height.</Text> : null}
    <Text style={styles.small}>Last skipped view: {demo.hint ?? 'none'}{demo.rejections ? ` / ${Object.entries(demo.rejections).map(([reason, count]) => `${reason}: ${count}`).join(', ')}` : ''}</Text>
    {demo.stage !== 'idle' ? <Text style={styles.copy}>
      {demo.views} views · {Math.round(demo.coverage * 100)}% coverage{demo.taps > 1 ? ` · ${demo.taps} taps` : ''}{demo.fitting ? ' · fitting…' : ''}
    </Text> : null}
    {rim ? <Text style={styles.copy}>Rim Ø {rim.diameterMm.toFixed(0)} mm · {rim.heightMm.toFixed(0)} mm high{food?.items.length ? '' : ` · ${demo.spots} probe spots 1 in inside`}</Text> : null}
    {demo.lidar && rim ? <Text style={styles.copy}>LiDAR check: Ø {demo.lidar.diameterMm.toFixed(0)} mm ({(rim.diameterMm - demo.lidar.diameterMm >= 0 ? '+' : '')}{(rim.diameterMm - demo.lidar.diameterMm).toFixed(0)} mm) · {demo.lidar.heightMm.toFixed(0)} mm high</Text>
      : demo.stage === 'done' && lidar && rim ? <Text style={styles.copy}>LiDAR check unavailable for this plate.</Text> : null}
    </View> : null}
    {food?.phase === 'working' ? <Text style={styles.copy}>Identifying the food{food.source === 'claude' ? ' with Claude' : ' on the phone'}…</Text> : null}
    {food?.phase === 'done' && food.dish ? <Text style={styles.copy}>Looks like {food.dish.label} ({Math.round(food.dish.confidence * 100)}%){food.dish.alternatives.length ? ` · or ${food.dish.alternatives.join(', ')}` : ''}</Text> : null}
    {details && food?.phase === 'done' && food.model ? <Text style={styles.small}>{food.model}{food.saved ? ` · saved scan ${food.saved}` : ''}</Text> : null}
    {food?.phase === 'failed' ? <Text style={styles.error}>{food.error}</Text> : null}
    {food?.items.map((item, i) => <View key={i} style={styles.legend}>
      <View style={[styles.swatch, { backgroundColor: item.color }]} />
      <Text style={styles.copy}>{item.label} · {item.category} · {item.points} point{item.points === 1 ? '' : 's'}{item.confidence < 0.6 ? ` · unsure (${Math.round(item.confidence * 100)}%)` : ''}{item.alternatives?.length && item.confidence < 0.6 ? ` · or ${item.alternatives.join(', ')}` : ''}</Text>
    </View>)}
    {ready && approved && food?.phase !== 'working' ? <Button title={food?.items.length ? 'Identify again' : 'Identify food'} onPress={() => identifyThenProbe(source)} /> : null}
    {food?.savedCount ? <Button title={`Share food scans (${food.savedCount})`} onPress={() => capture.shareFoodScans().catch(error => Alert.alert('Could not share', String(error)))} /> : null}
    {demo.probe?.running ? <Text style={styles.copy}>Probing {Math.min(demo.probe.done + 1, demo.probe.total)} of {demo.probe.total}…</Text>
      : fitted && approved && food?.phase !== 'working' ? <Button title={demo.probe?.done ? 'Probe again' : 'Simulate probing'} onPress={() => capture.demoProbe()} /> : null}
    {demo.spots > 0 ? <View style={styles.row}>
      <Button title="Points down 2 mm" onPress={() => capture.demoNudge(-2)} />
      <Text style={styles.copy}>{(demo.heightOffsetMm ?? 0) >= 0 ? '+' : ''}{(demo.heightOffsetMm ?? 0).toFixed(0)} mm</Text>
      <Button title="Up 2 mm" onPress={() => capture.demoNudge(2)} />
    </View> : null}
  </View>;
}
const styles = StyleSheet.create({
  page: { flex: 1, backgroundColor: brand.background },
  loading: { flex: 1, justifyContent: 'center', alignItems: 'center' },
  topLine: { height: 2, backgroundColor: brand.accent },
  header: { paddingHorizontal: 20, paddingVertical: 6 },
  headerRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 10 },
  logoBlock: { width: 100, gap: 6 },
  buildLabel: { fontFamily: brand.mono, fontSize: 8, letterSpacing: 1, color: brand.muted },
  eyebrow: { fontFamily: brand.mono, fontSize: 10, letterSpacing: 1, color: brand.body },
  tabs: { flex: 1, maxWidth: 245, flexDirection: 'row', borderBottomWidth: 1, borderColor: brand.line },
  tab: { flex: 1, minWidth: 0, minHeight: 44, paddingHorizontal: 5, alignItems: 'center', justifyContent: 'center', borderBottomWidth: 2, borderColor: 'transparent' },
  activeTab: { borderColor: brand.accent },
  tabLabel: { fontFamily: brand.sans, fontSize: 11, lineHeight: 14, textAlign: 'center', color: brand.muted },
  activeTabLabel: { color: brand.ink },
  cameraFrame: { flex: 1, minHeight: 220, marginHorizontal: 20, marginTop: 4, backgroundColor: brand.ink, overflow: 'hidden' },
  camera: { flex: 1 },
  cameraTag: { position: 'absolute', top: 12, left: 12, flexDirection: 'row', alignItems: 'center', gap: 7, padding: 8, backgroundColor: '#1A1A18CC' },
  liveDot: { width: 5, height: 5, backgroundColor: brand.accent },
  cameraTagText: { fontFamily: brand.mono, fontSize: 9, letterSpacing: 0.8, color: '#F1F0ED' },
  actionDock: { flexDirection: 'row', gap: 10, paddingHorizontal: 20, paddingVertical: 8, borderTopWidth: 1, borderColor: brand.line },
  controls: { flex: 1 }, controlsContent: { padding: 20, paddingBottom: 32 },
  copy: { paddingVertical: 6, color: brand.body, fontFamily: brand.sans, fontSize: 14, lineHeight: 20, flexShrink: 1 },
  small: { paddingVertical: 5, color: brand.muted, fontFamily: brand.mono, fontSize: 10, lineHeight: 16, flexShrink: 1 },
  error: { paddingVertical: 8, color: brand.danger, fontFamily: brand.sans, fontSize: 14, lineHeight: 20 },
  legend: { flexDirection: 'row', alignItems: 'center', gap: 8, borderBottomWidth: 1, borderColor: brand.line },
  swatch: { width: 7, height: 7 },
  headline: { paddingVertical: 8, fontSize: 18, lineHeight: 25, fontFamily: brand.sans, color: brand.ink },
  row: { flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center', justifyContent: 'space-between', gap: 8 },
  coverageCard: { borderTopWidth: 1, borderBottomWidth: 1, borderColor: brand.line, paddingVertical: 12, marginVertical: 8 },
  cellRow: { flexDirection: 'row', gap: 4, alignItems: 'center', marginVertical: 4 },
  cellLabel: { width: 42, fontFamily: brand.mono, fontSize: 9, color: brand.muted },
  cell: { flex: 1, height: 8, backgroundColor: brand.line },
  cellCovered: { backgroundColor: brand.accent },
  coverageNumber: { fontFamily: brand.mono, fontSize: 25, color: brand.ink },
  track: { height: 3, backgroundColor: brand.line, marginVertical: 12 },
  fill: { height: 3, backgroundColor: brand.accent },
});
