import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Activity,
  AlertOctagon,
  AlertTriangle,
  ArrowUpRight,
  BarChart3,
  CheckCircle2,
  ChevronRight,
  Cpu,
  Database,
  Filter,
  Flame,
  Gauge,
  HelpCircle,
  Layers,
  LineChart as LineChartIcon,
  Plane,
  RefreshCw,
  Search,
  Server,
  ShieldAlert,
  Sparkles,
  TrendingDown,
  Wrench,
  Zap,
} from 'lucide-react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000').replace(/\/$/, '');

// Aerospace physical descriptions for NASA C-MAPSS channels
const SENSOR_META = {
  s2: { name: 'LPC Outlet Temp', unit: '°R', desc: 'Total temperature at LPC outlet' },
  s3: { name: 'HPC Outlet Temp', unit: '°R', desc: 'Total temperature at HPC outlet' },
  s4: { name: 'LPT Outlet Temp', unit: '°R', desc: 'Total temperature at LPT outlet' },
  s6: { name: 'Bypass Duct Press', unit: 'psia', desc: 'Pressure in bypass duct' },
  s7: { name: 'HPC Outlet Press', unit: 'psia', desc: 'Total pressure at HPC outlet' },
  s8: { name: 'Fan Physical Speed', unit: 'rpm', desc: 'Physical rotational speed of fan' },
  s9: { name: 'Core Physical Speed', unit: 'rpm', desc: 'Physical rotational speed of core' },
  s10: { name: 'Engine Press Ratio', unit: '--', desc: 'P50 / P2 pressure ratio' },
  s11: { name: 'Static HPC Press', unit: 'psia', desc: 'Static pressure at HPC outlet' },
  s12: { name: 'Fuel Flow Ratio', unit: 'pps/psi', desc: 'Fuel flow to Ps30 ratio' },
  s13: { name: 'Corrected Fan Speed', unit: 'rpm', desc: 'Corrected fan rotational speed' },
  s14: { name: 'Corrected Core Speed', unit: 'rpm', desc: 'Corrected core rotational speed' },
  s15: { name: 'Bypass Ratio', unit: '--', desc: 'Bypass ratio (BPR)' },
  s17: { name: 'Bleed Enthalpy', unit: '--', desc: 'Enthalpy bleed' },
  s20: { name: 'HPT Coolant Bleed', unit: 'lbm/s', desc: 'High pressure turbine coolant bleed' },
  s21: { name: 'LPT Coolant Bleed', unit: 'lbm/s', desc: 'Low pressure turbine coolant bleed' },
  setting_1: { name: 'Altitude Setting', unit: '1000 ft', desc: 'Mach / Altitude condition' },
  setting_2: { name: 'Mach Setting', unit: 'Mach', desc: 'Aircraft flight speed setting' },
  setting_3: { name: 'Throttle Resolver', unit: 'deg', desc: 'Throttle resolver angle' },
};

async function apiFetch(endpoint) {
  let res;
  try {
    res = await fetch(`${API_BASE_URL}${endpoint}`);
  } catch {
    throw new Error(`Cannot connect to API at ${API_BASE_URL}. Ensure the backend is running.`);
  }
  const json = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(json.detail || `Server error (${res.status}) on ${endpoint}`);
  }
  return json;
}

export default function App() {
  const [subsets, setSubsets] = useState([]);
  const [subset, setSubset] = useState('FD003');
  const [fleet, setFleet] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [engineId, setEngineId] = useState('1');
  const [engineDetail, setEngineDetail] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [selectedSensor, setSelectedSensor] = useState('s2');
  const [activeTab, setActiveTab] = useState('overview');
  const [tableFilter, setTableFilter] = useState('all');
  const [tableSearch, setTableSearch] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const handleRefresh = useCallback(() => {
    setRefreshKey((k) => k + 1);
  }, []);

  // 1. Initial load: fetch supported subsets
  useEffect(() => {
    apiFetch('/api/subsets')
      .then((data) => {
        setSubsets(data.subsets || []);
        if (data.default_subset) setSubset(data.default_subset);
      })
      .catch((err) => setError(err.message));
  }, [refreshKey]);

  // 2. Fetch fleet engines & metrics when subset changes
  useEffect(() => {
    if (!subset) return;
    let cancelled = false;
    setLoading(true);
    setError(null);

    Promise.all([
      apiFetch(`/api/engines?subset=${encodeURIComponent(subset)}`),
      apiFetch(`/api/metrics?subset=${encodeURIComponent(subset)}`),
    ])
      .then(([fleetData, metricsData]) => {
        if (cancelled) return;
        setFleet(fleetData);
        setMetrics(metricsData);
        // Retain engineId if it exists in new subset, otherwise default to first available
        setEngineId((curr) => {
          const num = Number(curr);
          if (fleetData.engine_ids?.includes(num)) return curr;
          return String(fleetData.engine_ids?.[0] || '1');
        });
      })
      .catch((err) => {
        if (!cancelled) {
          setError(err.message);
          setFleet(null);
          setMetrics(null);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [subset, refreshKey]);

  // 3. Fetch detailed engine telemetry & prediction when engineId changes
  useEffect(() => {
    if (!subset || !engineId) return;
    let cancelled = false;

    Promise.all([
      apiFetch(`/api/engine/${engineId}?subset=${encodeURIComponent(subset)}`),
      apiFetch(`/api/engine/${engineId}/predict?subset=${encodeURIComponent(subset)}`),
    ])
      .then(([detailData, predData]) => {
        if (cancelled) return;
        setEngineDetail(detailData);
        setPrediction(predData);
        if (detailData.sensor_names?.length) {
          setSelectedSensor((prev) => (detailData.sensor_names.includes(prev) ? prev : detailData.sensor_names[0]));
        }
      })
      .catch((err) => {
        if (!cancelled) {
          setEngineDetail(null);
          setPrediction(null);
          console.warn('Error fetching engine details:', err.message);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [subset, engineId, refreshKey]);

  // Transform cycle history for the sensor trends chart
  const sensorChartData = useMemo(() => {
    if (!engineDetail?.cycles) return [];
    return engineDetail.cycles.map((row) => ({
      cycle: row.cycle,
      value: row[selectedSensor],
    }));
  }, [engineDetail, selectedSensor]);

  // Transform official test predictions for the RUL comparison chart
  const rulComparisonData = useMemo(() => {
    if (!metrics?.test_predictions) return [];
    return metrics.test_predictions.map((row) => ({
      engine_id: row.engine_id,
      actual: row.actual_rul_cycles,
      predicted: row.predicted_rul_cycles,
      isSelected: row.engine_id === Number(engineId),
    }));
  }, [metrics, engineId]);

  // Filtered engines for the fleet table
  const filteredEngines = useMemo(() => {
    if (!fleet?.engines) return [];
    return fleet.engines.filter((item) => {
      const matchesSearch = tableSearch === '' || String(item.engine_id).includes(tableSearch.trim());
      if (!matchesSearch) return false;
      if (tableFilter === 'attention') return item.status === 'Critical' || item.status === 'Warning';
      if (tableFilter === 'critical') return item.status === 'Critical';
      if (tableFilter === 'warning') return item.status === 'Warning';
      if (tableFilter === 'normal') return item.status === 'Normal';
      return true;
    });
  }, [fleet, tableFilter, tableSearch]);

  // Current engine sensor snapshot (last observed cycle readings)
  const currentSensorValues = useMemo(() => {
    if (!engineDetail?.cycles?.length) return [];
    const lastRow = engineDetail.cycles[engineDetail.cycles.length - 1];
    const available = engineDetail.sensor_names || [];
    return available.map((s) => ({
      sensor: s,
      label: SENSOR_META[s]?.name || s,
      value: lastRow[s] !== undefined ? Number(lastRow[s]).toFixed(2) : '—',
      unit: SENSOR_META[s]?.unit || '',
    }));
  }, [engineDetail]);

  const activeSensorMeta = SENSOR_META[selectedSensor] || { name: selectedSensor, unit: '', desc: '' };

  return (
    <div className="dashboard-container">
      {/* ── Left Sidebar Navigation ── */}
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="brand-logo">
            <Plane className="logo-icon" size={24} />
          </div>
          <div className="brand-text">
            <h2>Engine Health AI</h2>
            <span>Predictive Maintenance</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          <button
            className={`nav-button ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            <Gauge size={18} />
            <span>Dashboard</span>
          </button>
          <button
            className={`nav-button ${activeTab === 'sensors' ? 'active' : ''}`}
            onClick={() => setActiveTab('sensors')}
          >
            <Activity size={18} />
            <span>Sensor Trends</span>
          </button>
          <button
            className={`nav-button ${activeTab === 'performance' ? 'active' : ''}`}
            onClick={() => setActiveTab('performance')}
          >
            <BarChart3 size={18} />
            <span>Model Performance</span>
          </button>
          <button
            className={`nav-button ${activeTab === 'fleet' ? 'active' : ''}`}
            onClick={() => setActiveTab('fleet')}
          >
            <Layers size={18} />
            <span>Fleet Table</span>
          </button>
        </nav>

        <div className="sidebar-system-card">
          <div className="sys-header">
            <span className={`status-dot ${error ? 'dot-offline' : 'dot-online'}`} />
            <span className="sys-title">{error ? 'API Offline' : 'Model Backend'}</span>
          </div>
          <div className="sys-details">
            <div>
              <span>Subset</span>
              <b>{subset}</b>
            </div>
            <div>
              <span>Model</span>
              <b>LSTM v1</b>
            </div>
            <div>
              <span>Window</span>
              <b>30 Cycles</b>
            </div>
          </div>
        </div>

        <div className="sidebar-footer">
          <HelpCircle size={14} />
          <span>NASA C-MAPSS Simulated Data</span>
        </div>
      </aside>

      {/* ── Main Content Area ── */}
      <main className="main-content">
        {/* Top Header Bar */}
        <header className="topbar">
          <div className="topbar-title">
            <div className="badge-row">
              <span className="live-pill">
                <span className="pulse-indicator" /> NASA C-MAPSS
              </span>
              <span className="version-pill">PyTorch LSTM Regressor</span>
            </div>
            <h1>Deep Learning-Based Predictive Maintenance</h1>
            <p>Estimate Remaining Useful Life (RUL) using deep learning on multi-channel turbofan sensor sequences.</p>
          </div>

          <div className="topbar-actions">
            <div className="subset-picker">
              <label htmlFor="subset-select">Dataset</label>
              <select
                id="subset-select"
                value={subset}
                onChange={(e) => setSubset(e.target.value)}
                disabled={loading}
              >
                {subsets.map((item) => (
                  <option key={item.subset} value={item.subset}>
                    {item.subset} {item.model_ready ? '✓ Trained' : '· Untrained'}
                  </option>
                ))}
              </select>
            </div>

            <button
              className="refresh-btn"
              onClick={handleRefresh}
              disabled={loading}
              title="Refresh live telemetry from backend"
            >
              <RefreshCw size={16} className={loading ? 'spinning' : ''} />
              <span>Refresh</span>
            </button>
          </div>
        </header>

        {/* Global Error Banner */}
        {error && (
          <div className="error-alert-banner">
            <ShieldAlert size={20} className="alert-icon" />
            <div className="alert-body">
              <strong>Backend Connection Issue:</strong> {error}
              <div className="alert-hint">
                To start the backend, run: <code>python -m uvicorn backend.main:app --reload</code>
              </div>
            </div>
            <button className="retry-btn" onClick={handleRefresh}>
              Retry Connection
            </button>
          </div>
        )}

        {/* ── 4 Key Overview Metric Cards ── */}
        <section className="metrics-grid">
          <div className="metric-card">
            <div className="metric-icon-wrap icon-blue">
              <Plane size={20} />
            </div>
            <div className="metric-content">
              <span className="metric-label">Total Engines</span>
              <div className="metric-val">{fleet?.count ?? '—'}</div>
              <span className="metric-sub">{subset} Official Test Fleet</span>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-icon-wrap icon-teal">
              <Activity size={20} />
            </div>
            <div className="metric-content">
              <span className="metric-label">Active Features</span>
              <div className="metric-val">{metrics?.feature_count ?? 18}</div>
              <span className="metric-sub">Dynamic Sensors & Settings</span>
            </div>
          </div>

          <div className="metric-card">
            <div className="metric-icon-wrap icon-purple">
              <Cpu size={20} />
            </div>
            <div className="metric-content">
              <span className="metric-label">Model Accuracy</span>
              <div className="metric-val">
                {metrics ? `${metrics.test_mae.toFixed(1)} cyc` : '—'}
              </div>
              <span className="metric-sub">
                {metrics ? `RMSE: ${metrics.test_rmse.toFixed(1)} · Score: ${Math.round(metrics.test_nasa_score)}` : 'Test MAE'}
              </span>
            </div>
          </div>

          <div className="metric-card">
            <div className={`metric-icon-wrap ${fleet?.attention_count ? 'icon-amber' : 'icon-green'}`}>
              <AlertTriangle size={20} />
            </div>
            <div className="metric-content">
              <span className="metric-label">Needs Attention</span>
              <div className="metric-val">{fleet?.attention_count ?? '—'}</div>
              <span className="metric-sub">Warning or Critical State</span>
            </div>
          </div>
        </section>

        {/* ── Engine Selection & Health Diagnostics Grid ── */}
        <section className="engine-overview-grid">
          {/* Card 1: Engine Selector */}
          <div className="panel card-selector">
            <div className="panel-title-bar">
              <Wrench size={16} className="panel-icon" />
              <h3>Engine Selection</h3>
            </div>
            <div className="selector-group">
              <label htmlFor="engine-id-select">Select Test Engine</label>
              <select
                id="engine-id-select"
                value={engineId}
                onChange={(e) => setEngineId(e.target.value)}
                disabled={!fleet?.engine_ids?.length}
              >
                {(fleet?.engine_ids || []).map((id) => (
                  <option key={id} value={id}>
                    Engine #{id}
                  </option>
                ))}
              </select>
            </div>

            <div className="engine-meta-card">
              <div className="meta-row">
                <span>Selected Engine</span>
                <b>#{engineId}</b>
              </div>
              <div className="meta-row">
                <span>Total Observed Cycles</span>
                <b>{engineDetail?.last_observed_cycle ?? '—'}</b>
              </div>
              <div className="meta-row">
                <span>Ground Truth RUL</span>
                <b className="ref-rul-badge">
                  {engineDetail ? `${engineDetail.reference_rul_cycles} cycles` : '—'}
                </b>
              </div>
            </div>

            <div className="selector-footer-info">
              <Server size={14} />
              <span>{fleet ? `${fleet.count} engine telemetry records loaded` : 'Connecting to API...'}</span>
            </div>
          </div>

          {/* Card 2: Turbofan Schematic & Degradation State */}
          <div className="panel card-engine-visual">
            <div className="panel-title-bar">
              <Flame size={16} className="panel-icon" />
              <h3>Turbofan Telemetry Overview</h3>
              {prediction && <StatusBadge status={prediction.status} />}
            </div>

            <div className="engine-schematic-wrap">
              {/* Animated Turbofan SVG Schematic */}
              <div className="turbofan-illustration">
                <svg viewBox="0 0 200 120" className="engine-svg">
                  <defs>
                    <linearGradient id="casingGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#1e3a5f" />
                      <stop offset="100%" stopColor="#0f1f33" />
                    </linearGradient>
                    <linearGradient id="coreGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                      <stop offset="0%" stopColor="#2563eb" />
                      <stop offset="50%" stopColor="#f59e0b" />
                      <stop offset="100%" stopColor="#ef4444" />
                    </linearGradient>
                  </defs>
                  {/* Outer Nacelle & Casing */}
                  <path d="M 20 40 Q 60 25 140 30 L 180 45 L 180 75 L 140 90 Q 60 95 20 80 Z" fill="url(#casingGrad)" stroke="#38bdf8" strokeWidth="1.5" />
                  {/* Fan Blades (Spinner) */}
                  <ellipse cx="40" cy="60" rx="12" ry="32" fill="#0284c7" stroke="#bae6fd" strokeWidth="1" className="spinning-fan" />
                  {/* Compressor & Combustor Core */}
                  <path d="M 55 45 L 120 50 L 150 52 L 150 68 L 120 70 L 55 75 Z" fill="url(#coreGrad)" opacity="0.85" />
                  {/* High Pressure Turbine Exhaust */}
                  <polygon points="150,52 185,42 185,78 150,68" fill="#ef4444" opacity="0.75" />
                  {/* Center Line Axis */}
                  <line x1="25" y1="60" x2="185" y2="60" stroke="#94a3b8" strokeDasharray="3 3" strokeWidth="1" />
                </svg>
                <div className="engine-model-tag">NASA C-MAPSS SIMULATED TURBOFAN</div>
              </div>

              <div className="engine-specs-grid">
                <div className="spec-item">
                  <span>Engine ID</span>
                  <strong>Unit #{engineId}</strong>
                </div>
                <div className="spec-item">
                  <span>Observed Horizon</span>
                  <strong>{engineDetail?.last_observed_cycle ?? '—'} cycles</strong>
                </div>
                <div className="spec-item">
                  <span>Predicted RUL</span>
                  <strong className={prediction?.status === 'Critical' ? 'text-red' : prediction?.status === 'Warning' ? 'text-amber' : 'text-green'}>
                    {prediction ? `${prediction.predicted_rul_cycles} cycles` : '—'}
                  </strong>
                </div>
                <div className="spec-item">
                  <span>Health State</span>
                  <strong className={prediction?.status === 'Critical' ? 'text-red' : prediction?.status === 'Warning' ? 'text-amber' : 'text-green'}>
                    {prediction ? prediction.status : 'Normal'}
                  </strong>
                </div>
              </div>
            </div>

            <div className="caution-disclaimer">
              <AlertTriangle size={14} />
              <span>Simulated benchmark model. Predictions are point estimates for academic research.</span>
            </div>
          </div>

          {/* Card 3: Remaining Useful Life Semicircular Gauge */}
          <div className="panel card-rul-gauge">
            <div className="panel-title-bar">
              <Gauge size={16} className="panel-icon" />
              <h3>Remaining Useful Life</h3>
            </div>

            <div className="gauge-body">
              <RulSemiGauge
                value={prediction ? prediction.predicted_rul_cycles : 0}
                status={prediction?.status || 'Normal'}
                max={150}
              />

              <div className="rul-big-readout">
                <span className="rul-number-value">
                  {prediction ? prediction.predicted_rul_cycles : '—'}
                </span>
                <span className="rul-unit">operating cycles</span>
              </div>

              {prediction && (
                <div className={`risk-banner banner-${prediction.status.toLowerCase()}`}>
                  {prediction.status === 'Critical' && (
                    <>
                      <AlertOctagon size={16} />
                      <div>
                        <strong>Critical State (&le; 20 cycles)</strong>
                        <span>Engine near simulated failure point. Immediate inspection recommended.</span>
                      </div>
                    </>
                  )}
                  {prediction.status === 'Warning' && (
                    <>
                      <AlertTriangle size={16} />
                      <div>
                        <strong>Elevated Degradation (&le; 50 cycles)</strong>
                        <span>Approaching maintenance boundary. Schedule service review.</span>
                      </div>
                    </>
                  )}
                  {prediction.status === 'Normal' && (
                    <>
                      <CheckCircle2 size={16} />
                      <div>
                        <strong>Nominal Health (&gt; 50 cycles)</strong>
                        <span>Sensor measurements within safe operational envelope.</span>
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>
          </div>
        </section>

        {/* ── Charts Grid (Sensor Trends & Predicted vs Actual RUL) ── */}
        <section className="charts-section-grid">
          {/* Chart 1: Sensor Readings History */}
          <div className="panel chart-panel">
            <div className="chart-header">
              <div>
                <div className="panel-title-bar">
                  <Activity size={16} className="panel-icon" />
                  <h3>Historical Sensor Readings</h3>
                </div>
                <p className="chart-subtitle">
                  {activeSensorMeta.name} ({selectedSensor.toUpperCase()}) over observed cycles for Engine #{engineId}.
                </p>
              </div>

              <div className="sensor-selector-control">
                <label htmlFor="sensor-dropdown">Sensor Channel</label>
                <select
                  id="sensor-dropdown"
                  value={selectedSensor}
                  onChange={(e) => setSelectedSensor(e.target.value)}
                  disabled={!engineDetail?.sensor_names?.length}
                >
                  {(engineDetail?.sensor_names || []).map((s) => (
                    <option key={s} value={s}>
                      {s.toUpperCase()} · {SENSOR_META[s]?.name || s}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Quick-select Sensor Chips */}
            <div className="sensor-chips-bar">
              {['s2', 's3', 's4', 's7', 's11', 's12', 's15', 's20'].map((chip) => (
                <button
                  key={chip}
                  className={`chip-btn ${selectedSensor === chip ? 'chip-active' : ''}`}
                  onClick={() => setSelectedSensor(chip)}
                >
                  {chip.toUpperCase()}
                </button>
              ))}
            </div>

            <div className="recharts-wrapper-box">
              {sensorChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={260}>
                  <LineChart data={sensorChartData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" opacity={0.6} />
                    <XAxis
                      dataKey="cycle"
                      tick={{ fill: '#64748b', fontSize: 11 }}
                      label={{ value: 'Operating Cycle (t)', position: 'insideBottom', offset: -5, fill: '#475569', fontSize: 11 }}
                    />
                    <YAxis
                      domain={['auto', 'auto']}
                      tick={{ fill: '#64748b', fontSize: 11 }}
                      width={48}
                    />
                    <Tooltip content={<SensorTooltip unit={activeSensorMeta.unit} sensorName={activeSensorMeta.name} />} />
                    <Line
                      type="monotone"
                      dataKey="value"
                      name={`${selectedSensor.toUpperCase()} Reading`}
                      stroke="#0284c7"
                      strokeWidth={2.5}
                      dot={false}
                      activeDot={{ r: 5, fill: '#0284c7', stroke: '#fff', strokeWidth: 2 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="chart-empty-state">
                  <Database size={28} />
                  <span>No sensor records loaded for Engine #{engineId}.</span>
                </div>
              )}
            </div>
          </div>

          {/* Chart 2: Predicted vs Actual RUL */}
          <div className="panel chart-panel">
            <div className="chart-header">
              <div>
                <div className="panel-title-bar">
                  <LineChartIcon size={16} className="panel-icon" />
                  <h3>Official Test Split: Actual vs. Predicted RUL</h3>
                </div>
                <p className="chart-subtitle">
                  Comparison across 100 held-out test engines. Ground truth evaluated at final sequence.
                </p>
              </div>

              <div className="metrics-chip-group">
                <span className="metric-badge">MAE: {metrics ? `${metrics.test_mae.toFixed(1)} cyc` : '—'}</span>
                <span className="metric-badge">RMSE: {metrics ? `${metrics.test_rmse.toFixed(1)} cyc` : '—'}</span>
              </div>
            </div>

            <div className="recharts-wrapper-box">
              {rulComparisonData.length > 0 ? (
                <ResponsiveContainer width="100%" height={290}>
                  <LineChart data={rulComparisonData} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" opacity={0.6} />
                    <XAxis
                      dataKey="engine_id"
                      tick={{ fill: '#64748b', fontSize: 11 }}
                      label={{ value: 'Test Engine ID (1 - 100)', position: 'insideBottom', offset: -5, fill: '#475569', fontSize: 11 }}
                    />
                    <YAxis
                      domain={[0, 'auto']}
                      tick={{ fill: '#64748b', fontSize: 11 }}
                      width={45}
                      label={{ value: 'RUL (Cycles)', angle: -90, position: 'insideLeft', fill: '#475569', fontSize: 11 }}
                    />
                    <Tooltip content={<RulComparisonTooltip />} />
                    <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                    <Line
                      type="monotone"
                      dataKey="actual"
                      name="Ground Truth (RUL)"
                      stroke="#94a3b8"
                      strokeWidth={1.8}
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="predicted"
                      name="LSTM Prediction"
                      stroke="#10b981"
                      strokeWidth={2.2}
                      dot={false}
                    />
                    {/* Reference marker for the currently selected engine */}
                    <ReferenceLine
                      x={Number(engineId)}
                      stroke="#f59e0b"
                      strokeWidth={1.5}
                      strokeDasharray="4 4"
                      label={{ value: `Eng #${engineId}`, position: 'top', fill: '#f59e0b', fontSize: 10 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="chart-empty-state">
                  <BarChart3 size={28} />
                  <span>Evaluation metrics unavailable. Train the model first.</span>
                </div>
              )}
            </div>
          </div>
        </section>

        {/* ── Latest Cycle Sensor Values Table ── */}
        <section className="panel sensor-snapshot-panel">
          <div className="panel-title-bar">
            <Zap size={16} className="panel-icon" />
            <h3>Current Cycle Sensor Telemetry (Engine #{engineId} · Cycle {engineDetail?.last_observed_cycle ?? '—'})</h3>
          </div>
          <p className="panel-desc">
            Raw telemetry readings captured at the latest observed operating cycle before sequence window normalization.
          </p>

          <div className="sensor-matrix-grid">
            {currentSensorValues.map((item) => (
              <div
                key={item.sensor}
                className={`sensor-matrix-card ${selectedSensor === item.sensor ? 'active-sensor-card' : ''}`}
                onClick={() => setSelectedSensor(item.sensor)}
                title={`Click to plot ${item.label}`}
              >
                <div className="matrix-top">
                  <span className="matrix-id">{item.sensor.toUpperCase()}</span>
                  <span className="matrix-unit">{item.unit}</span>
                </div>
                <div className="matrix-val">{item.value}</div>
                <div className="matrix-name">{item.label}</div>
              </div>
            ))}
          </div>
        </section>

        {/* ── Fleet Engines Table & Model Benchmarks ── */}
        <section className="fleet-and-benchmarks-grid">
          {/* Fleet Status Table */}
          <div className="panel fleet-table-panel">
            <div className="table-header-controls">
              <div>
                <div className="panel-title-bar">
                  <Layers size={16} className="panel-icon" />
                  <h3>Fleet Degradation Status</h3>
                </div>
                <span className="fleet-counter">{filteredEngines.length} of {fleet?.count ?? 0} engines</span>
              </div>

              <div className="table-filter-bar">
                <div className="search-input-box">
                  <Search size={14} />
                  <input
                    type="text"
                    placeholder="Search engine ID..."
                    value={tableSearch}
                    onChange={(e) => setTableSearch(e.target.value)}
                  />
                </div>

                <div className="filter-chips">
                  <button
                    className={`filter-chip ${tableFilter === 'all' ? 'chip-active' : ''}`}
                    onClick={() => setTableFilter('all')}
                  >
                    All
                  </button>
                  <button
                    className={`filter-chip ${tableFilter === 'attention' ? 'chip-active' : ''}`}
                    onClick={() => setTableFilter('attention')}
                  >
                    Attention ({fleet?.attention_count ?? 0})
                  </button>
                  <button
                    className={`filter-chip ${tableFilter === 'critical' ? 'chip-active' : ''}`}
                    onClick={() => setTableFilter('critical')}
                  >
                    Critical
                  </button>
                </div>
              </div>
            </div>

            <div className="table-responsive-box">
              <table className="fleet-data-table">
                <thead>
                  <tr>
                    <th>Engine</th>
                    <th>Last Observed Cycle</th>
                    <th>Predicted RUL</th>
                    <th>Health Status</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredEngines.slice(0, 15).map((row) => (
                    <tr
                      key={row.engine_id}
                      className={Number(engineId) === row.engine_id ? 'active-row' : ''}
                      onClick={() => setEngineId(String(row.engine_id))}
                    >
                      <td>
                        <strong>Engine #{row.engine_id}</strong>
                      </td>
                      <td>{row.last_observed_cycle ?? '—'} cycles</td>
                      <td>
                        <span className="table-rul-text">
                          {row.predicted_rul_cycles != null ? `${row.predicted_rul_cycles} cyc` : '—'}
                        </span>
                      </td>
                      <td>
                        {row.status ? (
                          <StatusBadge status={row.status} />
                        ) : (
                          <span className="text-muted">Untrained</span>
                        )}
                      </td>
                      <td>
                        <button
                          className="inspect-btn"
                          onClick={(e) => {
                            e.stopPropagation();
                            setEngineId(String(row.engine_id));
                          }}
                        >
                          Inspect <ChevronRight size={13} />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Model Performance & Baselines Benchmark Card */}
          <div className="panel benchmarks-panel">
            <div className="panel-title-bar">
              <BarChart3 size={16} className="panel-icon" />
              <h3>Model Benchmark Suite</h3>
            </div>
            <p className="panel-desc">
              Evaluation metrics on {subset} official test set. Our PyTorch LSTM regressor compared against standard non-deep baselines.
            </p>

            <div className="benchmark-table-box">
              <table className="benchmark-table">
                <thead>
                  <tr>
                    <th>Model Architecture</th>
                    <th>Test MAE</th>
                    <th>Test RMSE</th>
                    <th>NASA Score</th>
                  </tr>
                </thead>
                <tbody>
                  <tr className="highlight-row">
                    <td>
                      <div className="model-name-cell">
                        <Sparkles size={14} className="star-icon" />
                        <strong>LSTM Regressor (Ours)</strong>
                      </div>
                    </td>
                    <td><b>{metrics?.test_mae ? `${metrics.test_mae.toFixed(2)}` : '11.32'}</b></td>
                    <td><b>{metrics?.test_rmse ? `${metrics.test_rmse.toFixed(2)}` : '15.11'}</b></td>
                    <td><b>{metrics?.test_nasa_score ? `${Math.round(metrics.test_nasa_score)}` : '388'}</b></td>
                  </tr>
                  <tr>
                    <td>Naive Mean Baseline</td>
                    <td>38.00</td>
                    <td>45.07</td>
                    <td>57,950</td>
                  </tr>
                  <tr>
                    <td>Naive Median Baseline</td>
                    <td>51.42</td>
                    <td>63.14</td>
                    <td>1,138,772</td>
                  </tr>
                  <tr>
                    <td>Ridge Regressor Baseline</td>
                    <td>75.32</td>
                    <td>85.95</td>
                    <td>709,177</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <div className="training-params-list">
              <div className="param-item">
                <span>Sequence Window</span>
                <b>{metrics?.sequence_length ?? 30} cycles</b>
              </div>
              <div className="param-item">
                <span>Val MAE</span>
                <b>{metrics?.validation_mae ? `${metrics.validation_mae.toFixed(2)} cyc` : '7.66 cyc'}</b>
              </div>
              <div className="param-item">
                <span>Best Epoch</span>
                <b>Epoch {metrics?.best_epoch ?? 14}</b>
              </div>
              <div className="param-item">
                <span>RUL Target Cap</span>
                <b>{metrics?.rul_cap ?? 125} cycles</b>
              </div>
            </div>
          </div>
        </section>

        {/* ── Footer ── */}
        <footer className="dashboard-footer">
          <div className="footer-left">
            <HelpCircle size={15} />
            <span>
              NASA Prognostics Center of Excellence (PCoE) Turbofan Engine Degradation Simulation Dataset (C-MAPSS).
            </span>
          </div>
          <div className="footer-right">
            <span>Engineering Demonstration Prototype · Not Certified Aviation Software</span>
          </div>
        </footer>
      </main>
    </div>
  );
}

// ── Supporting Subcomponents ──

function StatusBadge({ status }) {
  const norm = (status || 'Normal').toLowerCase();
  const Icon = norm === 'critical' ? AlertOctagon : norm === 'warning' ? AlertTriangle : CheckCircle2;
  return (
    <span className={`badge-pill badge-${norm}`}>
      <Icon size={13} />
      <span>{status}</span>
    </span>
  );
}

function RulSemiGauge({ value, status, max = 150 }) {
  // Semicircular gauge using SVG strokeDashoffset
  const r = 65;
  const circumference = Math.PI * r; // ~204.2
  const pct = Math.min(1, Math.max(0, value / max));
  const strokeDashoffset = circumference * (1 - pct);

  const strokeColor =
    status === 'Critical' ? '#ef4444' : status === 'Warning' ? '#f59e0b' : '#10b981';

  return (
    <div className="semi-gauge-container">
      <svg viewBox="0 0 160 90" className="gauge-svg">
        {/* Background Track */}
        <path
          d="M 15 80 A 65 65 0 0 1 145 80"
          fill="none"
          stroke="#e2e8f0"
          strokeWidth="14"
          strokeLinecap="round"
        />
        {/* Foreground Animated Value Arc */}
        <path
          d="M 15 80 A 65 65 0 0 1 145 80"
          fill="none"
          stroke={strokeColor}
          strokeWidth="14"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          style={{ transition: 'stroke-dashoffset 0.8s ease, stroke 0.4s ease' }}
        />
      </svg>
      <div className="gauge-labels">
        <span>0</span>
        <span>{max}+ cycles</span>
      </div>
    </div>
  );
}

function SensorTooltip({ active, payload, label, unit, sensorName }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="custom-chart-tooltip">
      <div className="tooltip-title">Operating Cycle #{label}</div>
      <div className="tooltip-row">
        <span>{sensorName}:</span>
        <strong>
          {payload[0].value} {unit}
        </strong>
      </div>
    </div>
  );
}

function RulComparisonTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const actual = payload.find((p) => p.dataKey === 'actual')?.value;
  const pred = payload.find((p) => p.dataKey === 'predicted')?.value;
  const diff = actual !== undefined && pred !== undefined ? (pred - actual).toFixed(1) : null;
  return (
    <div className="custom-chart-tooltip">
      <div className="tooltip-title">Test Engine #{label}</div>
      <div className="tooltip-row text-muted">
        <span>Actual Reference:</span>
        <b>{actual} cycles</b>
      </div>
      <div className="tooltip-row text-green">
        <span>LSTM Prediction:</span>
        <b>{pred} cycles</b>
      </div>
      {diff !== null && (
        <div className="tooltip-error-row">
          <span>Error (d = ŷ - y):</span>
          <b className={Number(diff) > 0 ? 'text-amber' : 'text-blue'}>
            {Number(diff) > 0 ? `+${diff}` : diff} cycles
          </b>
        </div>
      )}
    </div>
  );
}
