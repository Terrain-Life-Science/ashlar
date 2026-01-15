import React, { useState, useEffect } from 'react'
import SummaryCard from './components/SummaryCard'
import PerformanceMetrics from './components/PerformanceMetrics'
import AccuracyCharts from './components/AccuracyCharts'
import FileSizeMetrics from './components/FileSizeMetrics'
import CycleAccuracyCard from './components/CycleAccuracyCard'
import ScalingAnalysis from './components/ScalingAnalysis'
import './App.css'

function App() {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    // Load data from report.json
    fetch('/report.json')
      .then(response => {
        if (!response.ok) {
          throw new Error('Failed to load report.json')
        }
        return response.json()
      })
      .then(data => {
        setData(data)
        setLoading(false)
      })
      .catch(err => {
        setError(err.message)
        setLoading(false)
      })
  }, [])

  if (loading) {
    return (
      <div className="app">
        <header className="app-header">
          <h1>Ashlar Registration Visualizer</h1>
          <p>Registration pipeline performance and accuracy metrics</p>
        </header>
        <main className="app-main">
          <div className="loading">Loading registration data...</div>
        </main>
      </div>
    )
  }

  if (error) {
    return (
      <div className="app">
        <header className="app-header">
          <h1>Ashlar Registration Visualizer</h1>
          <p>Registration pipeline performance and accuracy metrics</p>
        </header>
        <main className="app-main">
          <div className="error">Error: {error}</div>
          <p>Make sure report.json is in the public directory or served by the API.</p>
        </main>
      </div>
    )
  }

  if (!data) {
    return (
      <div className="app">
        <header className="app-header">
          <h1>Ashlar Registration Visualizer</h1>
          <p>Registration pipeline performance and accuracy metrics</p>
        </header>
        <main className="app-main">
          <div className="error">No data available</div>
        </main>
      </div>
    )
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Ashlar Registration Visualizer</h1>
        <p>Registration pipeline performance and accuracy metrics</p>
      </header>
      <main className="app-main">
        {/* Check if this is a multi-scale report */}
        {data.runs && Array.isArray(data.runs) && data.runs.length > 0 ? (
          <>
            <ScalingAnalysis data={data} />
            {/* For multi-scale reports, show summary for each run or aggregated view */}
            <SummaryCard data={data} />
          </>
        ) : (
          <>
            {/* Single-run report - show all components */}
            <SummaryCard data={data} />
            <PerformanceMetrics data={data} />
            <FileSizeMetrics data={data} />
            <AccuracyCharts data={data} />
            <CycleAccuracyCard data={data} />
          </>
        )}
      </main>
    </div>
  )
}

export default App
