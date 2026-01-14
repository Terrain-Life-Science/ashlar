import React from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import './PerformanceMetrics.css'

function PerformanceMetrics({ data }) {
  if (!data || !data.performance) {
    return null
  }

  const { performance } = data

  // Prepare phase timing data
  const phaseTimingData = Object.entries(performance.phases || {})
    .map(([name, time]) => ({
      name: name.length > 20 ? name.substring(0, 20) + '...' : name,
      fullName: name,
      time: parseFloat(time.toFixed(2))
    }))
    .sort((a, b) => b.time - a.time)

  // Prepare memory data
  const memoryData = [
    { name: 'Peak Memory', value: parseFloat((performance.peak_memory_mb || 0).toFixed(2)) },
    { name: 'Current Memory', value: parseFloat((performance.current_memory_mb || 0).toFixed(2)) }
  ].filter(item => item.value > 0)

  // Prepare CPU utilization data
  const cpuData = [
    { name: 'CPU Cores', value: performance.cpu_cores_used || 0 },
    { name: 'CPU Threads', value: performance.cpu_threads_used || 0 }
  ]

  // Calculate total phase time
  const totalPhaseTime = phaseTimingData.reduce((sum, item) => sum + item.time, 0)

  // Colors for charts
  const COLORS = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe']

  return (
    <div className="performance-metrics">
      <h2>Performance Metrics</h2>
      
      <div className="metrics-grid">
        {/* Phase Timing Chart */}
        <div className="chart-container">
          <h3>Phase Timing</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={phaseTimingData} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis 
                dataKey="name" 
                angle={-45} 
                textAnchor="end" 
                height={100}
                interval={0}
              />
              <YAxis label={{ value: 'Time (seconds)', angle: -90, position: 'insideLeft' }} />
              <Tooltip 
                formatter={(value) => [`${value} seconds`, 'Time']}
                labelFormatter={(label) => {
                  const item = phaseTimingData.find(d => d.name === label)
                  return item ? item.fullName : label
                }}
              />
              <Legend />
              <Bar dataKey="time" fill="#667eea" name="Time (seconds)" />
            </BarChart>
          </ResponsiveContainer>
          <div className="chart-summary">
            <p>Total Phase Time: <strong>{totalPhaseTime.toFixed(2)} seconds</strong></p>
          </div>
        </div>

        {/* Memory Usage */}
        {memoryData.length > 0 && (
          <div className="chart-container">
            <h3>Memory Usage</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={memoryData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis label={{ value: 'Memory (MB)', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value) => [`${value} MB`, 'Memory']} />
                <Legend />
                <Bar dataKey="value" fill="#764ba2" name="Memory (MB)" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* CPU Utilization */}
        {cpuData[0].value > 0 && (
          <div className="chart-container">
            <h3>CPU Utilization</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={cpuData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis label={{ value: 'Count', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value) => [value, 'Count']} />
                <Legend />
                <Bar dataKey="value" fill="#4facfe" name="Count" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* GPU Status */}
        {performance.gpu_available !== undefined && (
          <div className="chart-container">
            <h3>GPU Status</h3>
            <div className="gpu-status">
              <div className={`status-indicator ${performance.gpu_available ? 'available' : 'unavailable'}`}>
                {performance.gpu_available ? '✓ Available' : '✗ Unavailable'}
              </div>
              {performance.gpu_available && performance.gpu_cores_used > 0 && (
                <p>GPU Cores Used: <strong>{performance.gpu_cores_used}</strong></p>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default PerformanceMetrics
