import React, { useState, useEffect } from 'react'
import SummaryCard from './components/SummaryCard'
import CycleAccuracyCard from './components/CycleAccuracyCard'
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
        <SummaryCard data={data} />
        <CycleAccuracyCard data={data} />
      </main>
    </div>
  )
}

export default App
