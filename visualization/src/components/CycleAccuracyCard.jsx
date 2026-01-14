import React from 'react'
import './CycleAccuracyCard.css'

function CycleAccuracyCard({ data }) {
  if (!data || !data.accuracy_by_cycle) {
    return null
  }

  return (
    <div className="cycle-accuracy-card">
      <h2>Registration Accuracy by Cycle</h2>
      <div className="cycles-grid">
        {data.accuracy_by_cycle.map((cycle) => (
          <div key={cycle.cycle_idx} className="cycle-item">
            <div className="cycle-header">
              <h3>Cycle {cycle.cycle_idx.toString().padStart(2, '0')}</h3>
              {cycle.cycle_idx === 0 && (
                <span className="badge badge-reference">Reference</span>
              )}
            </div>
            {cycle.cycle_idx === 0 ? (
              <div className="cycle-content">
                <p className="info-text">No alignment needed (reference cycle)</p>
              </div>
            ) : (
              <div className="cycle-content">
                <div className="metric-row">
                  <span className="metric-label">Coarse Shift:</span>
                  <span className="metric-value">
                    ({cycle.coarse_shift?.x?.toFixed(2) || '0.00'}, {cycle.coarse_shift?.y?.toFixed(2) || '0.00'}) px
                  </span>
                </div>
                <div className="metric-row">
                  <span className="metric-label">Coarse Error:</span>
                  <span className="metric-value">{cycle.coarse_error?.toFixed(4) || '0.0000'}</span>
                </div>
                {cycle.num_tiles > 0 && (
                  <>
                    <div className="metric-row">
                      <span className="metric-label">Tiles:</span>
                      <span className="metric-value">
                        {cycle.num_inliers || 0}/{cycle.num_tiles} inliers
                        ({((cycle.inlier_ratio || 0) * 100).toFixed(1)}%)
                      </span>
                    </div>
                    <div className="metric-row">
                      <span className="metric-label">RMSE:</span>
                      <span className="metric-value">{cycle.rmse?.toFixed(4) || '0.0000'} px</span>
                    </div>
                    {cycle.mean_residual > 0 && (
                      <div className="metric-row">
                        <span className="metric-label">Mean Residual:</span>
                        <span className="metric-value">{cycle.mean_residual?.toFixed(4) || '0.0000'} px</span>
                      </div>
                    )}
                  </>
                )}
                <div className="metric-row">
                  <span className="metric-label">Transform Type:</span>
                  <span className="metric-value badge badge-transform">{cycle.transform_type || 'N/A'}</span>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}

export default CycleAccuracyCard
