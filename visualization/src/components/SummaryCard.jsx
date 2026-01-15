import React from 'react'
import './SummaryCard.css'

function SummaryCard({ data }) {
  // Check if this is a multi-scale report
  const isMultiScale = data && data.runs && Array.isArray(data.runs) && data.runs.length > 0
  
  if (isMultiScale) {
    // For multi-scale reports, show summary of all runs
    const runs = data.runs
    return (
      <div className="summary-card">
        <h2>Multi-Scale Registration Summary</h2>
        <div className="summary-grid">
          <div className="summary-item">
            <span className="label">Number of Scales:</span>
            <span className="value">{runs.length}</span>
          </div>
          <div className="summary-item">
            <span className="label">Scale Factors:</span>
            <span className="value">{runs.map(r => `${r.scale_factor}x`).join(', ')}</span>
          </div>
          {runs.map(run => (
            <div key={run.scale_factor} className="summary-item">
              <span className="label">{run.scale_factor}x Scale ({run.image_size?.width}×{run.image_size?.height}):</span>
              <span className="value">{run.summary?.total_time_formatted || 'N/A'}</span>
            </div>
          ))}
        </div>
      </div>
    )
  }
  
  // Single-run report
  if (!data || !data.summary) {
    return null
  }

  const { summary, performance } = data

  return (
    <div className="summary-card">
      <h2>Registration Summary</h2>
      <div className="summary-grid">
        <div className="summary-item">
          <span className="label">Total Processing Time:</span>
          <span className="value">{summary.total_time_formatted || 'N/A'}</span>
        </div>
        <div className="summary-item">
          <span className="label">Number of Cycles:</span>
          <span className="value">{summary.num_cycles || 0}</span>
        </div>
        <div className="summary-item">
          <span className="label">Average RMSE:</span>
          <span className="value">{summary.average_rmse?.toFixed(4) || '0.0000'} pixels</span>
        </div>
        <div className="summary-item">
          <span className="label">Average Shift X:</span>
          <span className="value">{summary.average_shift_x?.toFixed(2) || '0.00'} ± {summary.average_std_shift_x?.toFixed(2) || '0.00'} pixels</span>
        </div>
        <div className="summary-item">
          <span className="label">Average Shift Y:</span>
          <span className="value">{summary.average_shift_y?.toFixed(2) || '0.00'} ± {summary.average_std_shift_y?.toFixed(2) || '0.00'} pixels</span>
        </div>
        {performance && (
          <>
            <div className="summary-item">
              <span className="label">CPU Cores Used:</span>
              <span className="value">{performance.cpu_cores_used || 'N/A'}</span>
            </div>
            <div className="summary-item">
              <span className="label">CPU Threads Used:</span>
              <span className="value">{performance.cpu_threads_used || 'N/A'}</span>
            </div>
            <div className="summary-item">
              <span className="label">Peak Memory:</span>
              <span className="value">{performance.peak_memory_mb?.toFixed(2) || '0.00'} MB</span>
            </div>
            {performance.total_input_size_mb !== undefined && (
              <div className="summary-item">
                <span className="label">Total Input Size:</span>
                <span className="value">{performance.total_input_size_mb.toFixed(2)} MB</span>
              </div>
            )}
            {performance.total_output_size_mb !== undefined && (
              <div className="summary-item">
                <span className="label">Total Output Size:</span>
                <span className="value">{performance.total_output_size_mb.toFixed(2)} MB</span>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default SummaryCard
