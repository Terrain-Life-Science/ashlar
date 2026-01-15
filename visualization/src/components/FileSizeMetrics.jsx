import React from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import './FileSizeMetrics.css'

function FileSizeMetrics({ data }) {
  // For multi-scale reports, this component is not shown (ScalingAnalysis handles it)
  if (data && data.runs && Array.isArray(data.runs) && data.runs.length > 0) {
    return null
  }
  
  if (!data || !data.performance) {
    return null
  }

  const { performance } = data

  // Prepare input file size data
  const inputFileData = Object.entries(performance.input_file_sizes_mb || {})
    .map(([path, size]) => {
      const fileName = path.split(/[/\\]/).pop() || path
      return {
        name: fileName.length > 30 ? fileName.substring(0, 30) + '...' : fileName,
        fullPath: path,
        size: parseFloat(size.toFixed(2))
      }
    })
    .sort((a, b) => b.size - a.size)

  // Prepare output file size data
  const outputFileData = Object.entries(performance.output_file_sizes_mb || {})
    .map(([path, size]) => {
      const fileName = path.split(/[/\\]/).pop() || path
      return {
        name: fileName.length > 30 ? fileName.substring(0, 30) + '...' : fileName,
        fullPath: path,
        size: parseFloat(size.toFixed(2))
      }
    })
    .sort((a, b) => b.size - a.size)

  // Prepare summary data
  const summaryData = [
    { name: 'Total Input', value: parseFloat((performance.total_input_size_mb || 0).toFixed(2)) },
    { name: 'Total Output', value: parseFloat((performance.total_output_size_mb || 0).toFixed(2)) }
  ].filter(item => item.value > 0)

  // Calculate compression ratio if both input and output exist
  const compressionRatio = summaryData.length === 2 && summaryData[0].value > 0
    ? parseFloat((summaryData[1].value / summaryData[0].value).toFixed(2))
    : null

  // Colors for charts
  const COLORS = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#00f2fe', '#43e97b']

  return (
    <div className="file-size-metrics">
      <h2>File Size & I/O Metrics</h2>
      
      <div className="metrics-grid">
        {/* Summary Pie Chart */}
        {summaryData.length > 0 && (
          <div className="chart-container">
            <h3>Total File Sizes</h3>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={summaryData}
                  cx="50%"
                  cy="50%"
                  labelLine={false}
                  label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(1)}%`}
                  outerRadius={80}
                  fill="#8884d8"
                  dataKey="value"
                >
                  {summaryData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => [`${value} MB`, 'Size']} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
            <div className="chart-summary">
              {summaryData.map(item => (
                <p key={item.name}>
                  {item.name}: <strong>{item.value} MB</strong>
                </p>
              ))}
              {compressionRatio !== null && (
                <p className="compression-info">
                  Output/Input Ratio: <strong>{compressionRatio}x</strong>
                </p>
              )}
            </div>
          </div>
        )}

        {/* Input File Sizes */}
        {inputFileData.length > 0 && (
          <div className="chart-container">
            <h3>Input File Sizes</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart 
                data={inputFileData} 
                margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="name" 
                  angle={-45} 
                  textAnchor="end" 
                  height={100}
                  interval={0}
                />
                <YAxis label={{ value: 'Size (MB)', angle: -90, position: 'insideLeft' }} />
                <Tooltip 
                  formatter={(value) => [`${value} MB`, 'Size']}
                  labelFormatter={(label) => {
                    const item = inputFileData.find(d => d.name === label)
                    return item ? item.fullPath : label
                  }}
                />
                <Legend />
                <Bar dataKey="size" fill="#667eea" name="File Size (MB)" />
              </BarChart>
            </ResponsiveContainer>
            <div className="chart-summary">
              <p>Total: <strong>{performance.total_input_size_mb?.toFixed(2) || '0.00'} MB</strong></p>
              <p>Files: <strong>{inputFileData.length}</strong></p>
            </div>
          </div>
        )}

        {/* Output File Sizes */}
        {outputFileData.length > 0 && (
          <div className="chart-container">
            <h3>Output File Sizes</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart 
                data={outputFileData} 
                margin={{ top: 20, right: 30, left: 20, bottom: 60 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis 
                  dataKey="name" 
                  angle={-45} 
                  textAnchor="end" 
                  height={100}
                  interval={0}
                />
                <YAxis label={{ value: 'Size (MB)', angle: -90, position: 'insideLeft' }} />
                <Tooltip 
                  formatter={(value) => [`${value} MB`, 'Size']}
                  labelFormatter={(label) => {
                    const item = outputFileData.find(d => d.name === label)
                    return item ? item.fullPath : label
                  }}
                />
                <Legend />
                <Bar dataKey="size" fill="#764ba2" name="File Size (MB)" />
              </BarChart>
            </ResponsiveContainer>
            <div className="chart-summary">
              <p>Total: <strong>{performance.total_output_size_mb?.toFixed(2) || '0.00'} MB</strong></p>
              <p>Files: <strong>{outputFileData.length}</strong></p>
            </div>
          </div>
        )}

        {/* Comparison Chart */}
        {inputFileData.length > 0 && outputFileData.length > 0 && (
          <div className="chart-container">
            <h3>Input vs Output Comparison</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart 
                data={[
                  { name: 'Total Input', size: performance.total_input_size_mb || 0 },
                  { name: 'Total Output', size: performance.total_output_size_mb || 0 }
                ]}
                margin={{ top: 20, right: 30, left: 20, bottom: 20 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis label={{ value: 'Size (MB)', angle: -90, position: 'insideLeft' }} />
                <Tooltip formatter={(value) => [`${value} MB`, 'Size']} />
                <Legend />
                <Bar dataKey="size" fill="#4facfe" name="Total Size (MB)" />
              </BarChart>
            </ResponsiveContainer>
            {compressionRatio !== null && (
              <div className="chart-summary">
                <p className="compression-info">
                  <strong>Compression Ratio:</strong> {compressionRatio}x
                  {compressionRatio < 1 ? ' (compressed)' : compressionRatio > 1 ? ' (expanded)' : ' (same size)'}
                </p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}

export default FileSizeMetrics
