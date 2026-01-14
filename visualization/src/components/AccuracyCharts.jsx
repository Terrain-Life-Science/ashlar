import React from 'react'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, ZAxis } from 'recharts'
import './AccuracyCharts.css'

function AccuracyCharts({ data }) {
  if (!data || !data.accuracy_by_cycle) {
    return null
  }

  const cycles = data.accuracy_by_cycle || []

  // Filter out reference cycle (cycle 0) for most charts
  const nonRefCycles = cycles.filter(c => c.cycle_idx !== 0)

  // Prepare shift magnitude data
  const shiftData = nonRefCycles.map(cycle => ({
    cycle: `Cycle ${cycle.cycle_idx.toString().padStart(2, '0')}`,
    cycleIdx: cycle.cycle_idx,
    shiftX: parseFloat((cycle.coarse_shift?.x || 0).toFixed(2)),
    shiftY: parseFloat((cycle.coarse_shift?.y || 0).toFixed(2)),
    shiftMagnitude: parseFloat(Math.sqrt(
      Math.pow(cycle.coarse_shift?.x || 0, 2) + 
      Math.pow(cycle.coarse_shift?.y || 0, 2)
    ).toFixed(2))
  }))

  // Prepare error data
  const errorData = nonRefCycles.map(cycle => ({
    cycle: `Cycle ${cycle.cycle_idx.toString().padStart(2, '0')}`,
    cycleIdx: cycle.cycle_idx,
    coarseError: parseFloat((cycle.coarse_error || 0).toFixed(4)),
    rmse: parseFloat((cycle.rmse || 0).toFixed(4)),
    meanResidual: parseFloat((cycle.mean_residual || 0).toFixed(4))
  }))

  // Prepare scatter plot data for shift vectors
  const shiftVectorData = nonRefCycles.map(cycle => ({
    x: parseFloat((cycle.coarse_shift?.x || 0).toFixed(2)),
    y: parseFloat((cycle.coarse_shift?.y || 0).toFixed(2)),
    cycle: `Cycle ${cycle.cycle_idx.toString().padStart(2, '0')}`,
    error: parseFloat((cycle.coarse_error || 0).toFixed(4))
  }))

  // Prepare inlier ratio data
  const inlierData = nonRefCycles
    .filter(cycle => cycle.num_tiles > 0)
    .map(cycle => ({
      cycle: `Cycle ${cycle.cycle_idx.toString().padStart(2, '0')}`,
      cycleIdx: cycle.cycle_idx,
      inlierRatio: parseFloat(((cycle.inlier_ratio || 0) * 100).toFixed(1)),
      numTiles: cycle.num_tiles,
      numInliers: cycle.num_inliers
    }))

  return (
    <div className="accuracy-charts">
      <h2>Registration Accuracy Analysis</h2>
      
      <div className="charts-grid">
        {/* Shift Magnitude Chart */}
        {shiftData.length > 0 && (
          <div className="chart-container">
            <h3>Coarse Shift Magnitude</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={shiftData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="cycle" />
                <YAxis label={{ value: 'Shift (pixels)', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value) => [`${value} pixels`, 'Magnitude']} />
                <Legend />
                <Bar dataKey="shiftMagnitude" fill="#667eea" name="Shift Magnitude (px)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Shift Components Chart */}
        {shiftData.length > 0 && (
          <div className="chart-container">
            <h3>Coarse Shift Components</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={shiftData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="cycle" />
                <YAxis label={{ value: 'Shift (pixels)', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value) => [`${value} pixels`, 'Shift']} />
                <Legend />
                <Bar dataKey="shiftX" fill="#f093fb" name="Shift X (px)" />
                <Bar dataKey="shiftY" fill="#4facfe" name="Shift Y (px)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Error Metrics Chart */}
        {errorData.length > 0 && (
          <div className="chart-container">
            <h3>Registration Errors</h3>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={errorData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="cycle" />
                <YAxis label={{ value: 'Error (pixels)', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value) => [`${value} pixels`, 'Error']} />
                <Legend />
                <Line type="monotone" dataKey="coarseError" stroke="#f44336" name="Coarse Error" strokeWidth={2} />
                <Line type="monotone" dataKey="rmse" stroke="#ff9800" name="RMSE" strokeWidth={2} />
                {errorData.some(d => d.meanResidual > 0) && (
                  <Line type="monotone" dataKey="meanResidual" stroke="#4caf50" name="Mean Residual" strokeWidth={2} />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Shift Vector Scatter Plot */}
        {shiftVectorData.length > 0 && (
          <div className="chart-container">
            <h3>Shift Vector Visualization</h3>
            <ResponsiveContainer width="100%" height={300}>
              <ScatterChart margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  type="number" 
                  dataKey="x" 
                  name="Shift X"
                  label={{ value: 'Shift X (pixels)', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  type="number" 
                  dataKey="y" 
                  name="Shift Y"
                  label={{ value: 'Shift Y (pixels)', angle: -90, position: 'insideLeft' }}
                />
                <ZAxis type="number" dataKey="error" range={[50, 500]} name="Error" />
                <Tooltip 
                  cursor={{ strokeDasharray: '3 3' }}
                  formatter={(value, name) => {
                    if (name === 'Error') return [`${value} pixels`, 'Error']
                    return [`${value} pixels`, name]
                  }}
                  labelFormatter={(label) => {
                    const item = shiftVectorData.find(d => d.x === label)
                    return item ? item.cycle : label
                  }}
                />
                <Legend />
                <Scatter name="Cycles" data={shiftVectorData} fill="#667eea">
                  {shiftVectorData.map((entry, index) => (
                    <text
                      x={entry.x}
                      y={entry.y}
                      textAnchor="middle"
                      fill="#333"
                      fontSize={10}
                      key={index}
                    >
                      {entry.cycle.split(' ')[1]}
                    </text>
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Inlier Ratio Chart */}
        {inlierData.length > 0 && (
          <div className="chart-container">
            <h3>Inlier Ratio (Fine Registration)</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={inlierData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="cycle" />
                <YAxis label={{ value: 'Inlier Ratio (%)', angle: -90, position: 'insideLeft' }} />
                <Tooltip 
                  formatter={(value, name) => {
                    if (name === 'Inlier Ratio') return [`${value}%`, 'Inlier Ratio']
                    return [value, name]
                  }}
                />
                <Legend />
                <Bar dataKey="inlierRatio" fill="#4caf50" name="Inlier Ratio (%)" />
              </BarChart>
            </ResponsiveContainer>
            <div className="chart-summary">
              {inlierData.map(item => (
                <p key={item.cycleIdx}>
                  {item.cycle}: <strong>{item.numInliers}/{item.numTiles}</strong> tiles
                </p>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default AccuracyCharts
