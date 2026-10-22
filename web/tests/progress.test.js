import test from 'node:test';
import assert from 'node:assert/strict';
import { progressDisplay, duration } from '../progress.js';

test('legacy milestones never appear as measured progress', () => {
  const view = progressDisplay({status: 'running', stage: 'dense_stereo', percent: 65, stage_percent: null});
  assert.equal(view.percent, null);
  assert.equal(view.indeterminate, true);
});

test('stage counts, timing and output age are displayed', () => {
  const view = progressDisplay({status: 'running', stage_percent: 50, completed_units: 139, total_units: 278,
    unit: 'depth maps', work_detail: 'Photometric: 139/139 · Geometric consistency: 0/139',
    elapsed_seconds: 1900, stage_elapsed_seconds: 1200, last_output_seconds_ago: 5});
  assert.equal(view.percent, 50);
  assert.match(view.label, /139 \/ 278 depth maps/);
  assert.match(view.detail, /Geometric consistency: 0\/139/);
  assert.match(view.timing, /31m 40s/);
  assert.match(view.timing, /5s ago/);
  assert.match(view.note, /not total run time/);
});

test('zero completed outputs is a measured zero, not unknown', () => {
  assert.equal(progressDisplay({status: 'running', stage_percent: 0}).indeterminate, false);
});

test('terminal results stop activity animation', () => {
  assert.equal(progressDisplay({status: 'done'}).percent, 100);
  assert.equal(progressDisplay({status: 'error'}).indeterminate, false);
});

test('long durations are readable', () => {
  assert.equal(duration(3601), '1h 0m');
  assert.equal(duration(-3), '0s');
});
