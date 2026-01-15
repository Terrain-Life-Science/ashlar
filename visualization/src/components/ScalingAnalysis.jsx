import React from 'react'
import { LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import './ScalingAnalysis.css'

function ScalingAnalysis({ data }) {
  // Check if this is a multi-scale report
  if (!data || !data.runs || !Array.isArray(data.runs) || data.runs.length === 0) {
    return null
  }

  const runs = data.runs

  // Prepare data for scaling charts
  const scalingData = runs.map(run => ({
    scale: `${run.scale_factor}x`,
    scaleFactor: run.scale_factor,
    width: run.image_size?.width || 0,
    height: run.image_size?.height || 0,
    totalTime: run.summary?.total_time_seconds || 0,
    totalTimeFormatted: run.summary?.total_time_formatted || '0s',
    peakMemory: run.performance?.peak_memory_mb || 0,
    totalInputSize: run.performance?.total_input_size_mb || 0,
    totalOutputSize: run.performance?.total_output_size_mb || 0,
    averageRMSE: run.summary?.average_rmse || 0,
    cpuCores: run.performance?.cpu_cores_used || 0,
    cpuThreads: run.performance?.cpu_threads_used || 0,
  })).sort((a, b) => a.scaleFactor - b.scaleFactor)

  // Calculate scaling ratios (relative to 1x)
  const baseRun = scalingData.find(r => r.scaleFactor === 1.0)
  if (baseRun) {
    scalingData.forEach(run => {
      run.timeRatio = baseRun.totalTime > 0 ? (run.totalTime / baseRun.totalTime).toFixed(2) : 0
      run.memoryRatio = baseRun.peakMemory > 0 ? (run.peakMemory / baseRun.peakMemory).toFixed(2) : 0
      run.sizeRatio = baseRun.totalInputSize > 0 ? (run.totalInputSize / baseRun.totalInputSize).toFixed(2) : 0
    })
  }

  // Prepare phase timing data across scales
  const phaseNames = new Set()
  runs.forEach(run => {
    if (run.performance?.phases) {
      Object.keys(run.performance.phases).forEach(phase => phaseNames.add(phase))
    }
  })
  
  const phaseData = Array.from(phaseNames).map(phaseName => {
    const phaseDataPoint = { phase: phaseName }
    runs.forEach(run => {
      const scaleLabel = `${run.scale_factor}x`
      phaseDataPoint[scaleLabel] = run.performance?.phases?.[phaseName] || 0
    })
    return phaseDataPoint
  })

  // Colors for different scales
  const scaleColors = {
    '1x': '#667eea',
    '2x': '#764ba2',
    '4x': '#f093fb'
  }

  return (
    <div className="scaling-analysis">
      <h2>Scaling Analysis</h2>
      <p className="scaling-description">
        Performance metrics across different image scales. All metrics are compared relative to the 1x baseline.
      </p>

      <div className="scaling-grid">
        {/* Processing Time vs Scale */}
        <div className="chart-container">
          <h3>Processing Time vs Scale</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={scalingData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="scale" 
                label={{ value: 'Scale Factor', position: 'top', offset: 10 }}
              />
              <YAxis 
                label={{ value: 'Time (seconds)', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip 
                formatter={(value) => [`${parseFloat(value).toFixed(2)}s`, 'Time']}
                labelFormatter={(label) => `Scale: ${label}`}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="totalTime" 
                stroke={scaleColors['1x']} 
                strokeWidth={2}
                name="Total Time (s)"
                dot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
          <div className="chart-summary">
            {scalingData.map(run => (
              <p key={run.scale}>
                <strong>{run.scale}:</strong> {run.totalTimeFormatted} 
                {run.timeRatio && run.scaleFactor !== 1.0 && ` (${run.timeRatio}x baseline)`}
              </p>
            ))}
          </div>
        </div>

        {/* Memory Usage vs Scale */}
        <div className="chart-container">
          <h3>Peak Memory Usage vs Scale</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={scalingData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="scale" 
                label={{ value: 'Scale Factor', position: 'top', offset: 10 }}
              />
              <YAxis 
                label={{ value: 'Memory (MB)', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip 
                formatter={(value) => [`${parseFloat(value).toFixed(2)} MB`, 'Peak Memory']}
                labelFormatter={(label) => `Scale: ${label}`}
              />
              <Legend />
              <Bar dataKey="peakMemory" fill={scaleColors['2x']} name="Peak Memory (MB)" />
            </BarChart>
          </ResponsiveContainer>
          <div className="chart-summary">
            {scalingData.map(run => (
              <p key={run.scale}>
                <strong>{run.scale}:</strong> {run.peakMemory.toFixed(2)} MB
                {run.memoryRatio && run.scaleFactor !== 1.0 && ` (${run.memoryRatio}x baseline)`}
              </p>
            ))}
          </div>
        </div>

        {/* File Sizes vs Scale */}
        <div className="chart-container">
          <h3>File Sizes vs Scale</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={scalingData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="scale" 
                label={{ value: 'Scale Factor', position: 'top', offset: 10 }}
              />
              <YAxis 
                label={{ value: 'Size (MB)', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip 
                formatter={(value) => [`${parseFloat(value).toFixed(2)} MB`, '']}
                labelFormatter={(label) => `Scale: ${label}`}
              />
              <Legend />
              <Bar dataKey="totalInputSize" fill={scaleColors['1x']} name="Total Input (MB)" />
              <Bar dataKey="totalOutputSize" fill={scaleColors['4x']} name="Total Output (MB)" />
            </BarChart>
          </ResponsiveContainer>
          <div className="chart-summary">
            {scalingData.map(run => (
              <p key={run.scale}>
                <strong>{run.scale}:</strong> Input: {run.totalInputSize.toFixed(2)} MB, 
                Output: {run.totalOutputSize.toFixed(2)} MB
              </p>
            ))}
          </div>
        </div>

        {/* Accuracy (RMSE) vs Scale */}
        <div className="chart-container">
          <h3>Registration Accuracy (RMSE) vs Scale</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={scalingData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="scale" 
                label={{ value: 'Scale Factor', position: 'top', offset: 10 }}
              />
              <YAxis 
                label={{ value: 'RMSE (pixels)', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip 
                formatter={(value) => [`${parseFloat(value).toFixed(4)} px`, 'RMSE']}
                labelFormatter={(label) => `Scale: ${label}`}
              />
              <Legend />
              <Line 
                type="monotone" 
                dataKey="averageRMSE" 
                stroke={scaleColors['4x']} 
                strokeWidth={2}
                name="Average RMSE (px)"
                dot={{ r: 6 }}
              />
            </LineChart>
          </ResponsiveContainer>
          <div className="chart-summary">
            {scalingData.map(run => (
              <p key={run.scale}>
                <strong>{run.scale}:</strong> {run.averageRMSE.toFixed(4)} pixels
              </p>
            ))}
          </div>
        </div>

        {/* Phase Timing Comparison */}
        {phaseData.length > 0 && (
          <div className="chart-container chart-container-wide">
            <h3>Phase Timing Comparison Across Scales</h3>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={phaseData} margin={{ top: 20, right: 30, left: 20, bottom: 80 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="phase" 
                  angle={-45}
                  textAnchor="end"
                  height={100}
                  label={{ value: 'Phase', position: 'insideBottom', offset: -5 }}
                />
                <YAxis 
                  label={{ value: 'Time (seconds)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip 
                  formatter={(value) => [`${parseFloat(value).toFixed(2)}s`, '']}
                />
                <Legend />
                {runs.map(run => {
                  const scaleLabel = `${run.scale_factor}x`
                  return (
                    <Bar 
                      key={scaleLabel}
                      dataKey={scaleLabel} 
                      fill={scaleColors[scaleLabel] || '#888'} 
                      name={`${scaleLabel} Scale`}
                    />
                  )
                })}
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>
    </div>
  )
}

export default ScalingAnalysis
