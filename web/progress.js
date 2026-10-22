export function duration(seconds) {
  const value = Math.max(0, Math.floor(seconds));
  if (value >= 3600) return `${Math.floor(value / 3600)}h ${Math.floor(value % 3600 / 60)}m`;
  if (value >= 60) return `${Math.floor(value / 60)}m ${value % 60}s`;
  return `${value}s`;
}

export function progressDisplay(job) {
  const active = job.status === 'running' || job.status === 'queued';
  // Legacy job.percent is a stage milestone, not a measured percentage.
  const measured = Number.isFinite(job.stage_percent);
  const percent = job.status === 'done' ? 100 : measured ? Math.max(0, Math.min(100, job.stage_percent)) : null;
  const label = measured
    ? `${job.completed_units} / ${job.total_units} ${job.unit} written · ${percent}%`
    : job.status === 'done' ? 'Complete' : active ? 'Working — percentage unavailable for this stage' : '';
  const timing = [];
  if (Number.isFinite(job.elapsed_seconds)) timing.push(`Total elapsed: ${duration(job.elapsed_seconds)}`);
  if (active && Number.isFinite(job.stage_elapsed_seconds)) timing.push(`This stage: ${duration(job.stage_elapsed_seconds)}`);
  if (active && Number.isFinite(job.last_output_seconds_ago)) timing.push(`Last completed map: ${duration(job.last_output_seconds_ago)} ago`);
  return {
    percent, label, timing: timing.join(' · '), detail: job.work_detail || '',
    indeterminate: active && percent === null,
    note: measured
      ? 'Progress counts depth-map outputs for this stage, not total run time. Fusion, cleanup, meshing and export follow.'
      : active ? 'This stage does not expose a reliable completion count. Its elapsed time is shown when available.' : '',
  };
}
