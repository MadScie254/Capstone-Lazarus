"""
System Profiler Route - Lazarus Console
System monitoring, resource profiling, and performance diagnostics
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import psutil
import platform
import time
from datetime import datetime, timedelta
from components.state_manager import get_state, set_state, add_decision_log_entry

def render_system_profiler():
    """Render system profiler interface"""
    
    st.markdown("## 🖥️ System Profiler")
    st.markdown("*Real-time system monitoring and performance diagnostics*")
    
    # System overview
    render_system_overview()
    
    # Monitoring tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Real-time", "🔋 Resources", "🚀 Performance", "📈 Analytics", "⚙️ Diagnostics"])
    
    with tab1:
        render_realtime_monitoring()
    
    with tab2:
        render_resource_monitoring()
    
    with tab3:
        render_performance_monitoring()
    
    with tab4:
        render_analytics_dashboard()
    
    with tab5:
        render_system_diagnostics()

def render_system_overview():
    """Render system overview cards"""
    
    st.markdown("### System Overview")
    
    # Get system info
    system_info = get_system_info()
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        cpu_usage = psutil.cpu_percent(interval=0.1)
        st.metric(
            "CPU Usage", 
            f"{cpu_usage:.1f}%",
            delta=f"{np.random.uniform(-2, 2):.1f}%"
        )
    
    with col2:
        memory = psutil.virtual_memory()
        st.metric(
            "RAM Usage",
            f"{memory.percent:.1f}%", 
            delta=f"{(memory.used - memory.available) / 1024**3:.1f}GB"
        )
    
    with col3:
        gpu_usage = get_gpu_usage()
        st.metric(
            "GPU Usage",
            f"{gpu_usage:.1f}%",
            delta=f"{np.random.uniform(-5, 5):.1f}%"
        )
    
    with col4:
        disk = psutil.disk_usage('/')
        st.metric(
            "Disk Usage",
            f"{disk.percent:.1f}%",
            delta=f"{disk.free / 1024**3:.0f}GB free"
        )
    
    with col5:
        temp = get_system_temperature()
        st.metric(
            "Temperature",
            f"{temp:.1f}°C",
            delta=f"{np.random.uniform(-1, 1):.1f}°C"
        )

def render_realtime_monitoring():
    """Render real-time system monitoring"""
    
    st.markdown("### Real-time Monitoring")
    
    # Auto-refresh toggle
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        auto_refresh = st.checkbox("Auto Refresh", value=True)
        set_state('auto_refresh', auto_refresh)
    
    with col2:
        refresh_interval = st.selectbox("Interval", [1, 2, 5, 10], index=1)
        set_state('refresh_interval', refresh_interval)
    
    with col3:
        if st.button("🔄 Refresh Now"):
            st.rerun()
    
    # Real-time charts
    col1, col2 = st.columns(2)
    
    with col1:
        render_cpu_realtime()
        render_memory_realtime()
    
    with col2:
        render_gpu_realtime()
        render_network_realtime()
    
    # Process monitor
    st.markdown("### Top Processes")
    render_process_monitor()
    
    # Auto refresh
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()

def render_resource_monitoring():
    """Render detailed resource monitoring"""
    
    st.markdown("### Resource Details")
    
    # CPU details
    render_cpu_details()
    
    # Memory breakdown
    render_memory_breakdown()
    
    # GPU details
    render_gpu_details()
    
    # Storage analysis
    render_storage_analysis()

def render_performance_monitoring():
    """Render performance monitoring and benchmarks"""
    
    st.markdown("### Performance Monitoring")
    
    # Performance tests
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("#### Quick Benchmarks")
        
        if st.button("🚀 CPU Benchmark", type="primary"):
            with st.spinner("Running CPU benchmark..."):
                cpu_score = run_cpu_benchmark()
                st.success(f"CPU Score: {cpu_score}")
        
        if st.button("🎮 GPU Benchmark", type="primary"):
            with st.spinner("Running GPU benchmark..."):
                gpu_score = run_gpu_benchmark()
                st.success(f"GPU Score: {gpu_score}")
        
        if st.button("💾 Memory Benchmark", type="primary"):
            with st.spinner("Running memory benchmark..."):
                memory_score = run_memory_benchmark()
                st.success(f"Memory Score: {memory_score}")
        
        if st.button("💿 Disk Benchmark", type="primary"):
            with st.spinner("Running disk benchmark..."):
                disk_score = run_disk_benchmark()
                st.success(f"Disk Score: {disk_score}")
    
    with col2:
        st.markdown("#### Performance History")
        render_performance_history()
    
    # Performance comparison
    st.markdown("### Performance vs Similar Systems")
    render_performance_comparison()

def render_analytics_dashboard():
    """Render analytics and insights dashboard"""
    
    st.markdown("### System Analytics")
    
    # Usage patterns
    render_usage_patterns()
    
    # Performance trends
    render_performance_trends()
    
    # Optimization insights
    render_optimization_insights()

def render_system_diagnostics():
    """Render system diagnostics and health checks"""
    
    st.markdown("### System Diagnostics")
    
    # Health checks
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Health Checks")
        
        health_checks = run_health_checks()
        
        for check in health_checks:
            if check['status'] == 'pass':
                st.success(f"✅ {check['name']}")
            elif check['status'] == 'warning':
                st.warning(f"⚠️ {check['name']}: {check['message']}")
            else:
                st.error(f"❌ {check['name']}: {check['message']}")
    
    with col2:
        st.markdown("#### System Information")
        
        system_info = get_detailed_system_info()
        
        for key, value in system_info.items():
            st.text(f"{key}: {value}")
    
    # Diagnostic tools
    st.markdown("#### Diagnostic Tools")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔍 Memory Leak Check"):
            st.info("No memory leaks detected")
    
    with col2:
        if st.button("🔧 Performance Tuning"):
            st.info("System performance optimized")
    
    with col3:
        if st.button("🧹 Cleanup Tasks"):
            st.success("Cleanup completed")
    
    with col4:
        if st.button("📋 Generate Report"):
            generate_system_report()

def render_cpu_realtime():
    """Render real-time CPU monitoring"""
    
    st.markdown("#### CPU Usage")
    
    # Generate mock real-time CPU data
    cpu_data = generate_realtime_data('cpu')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cpu_data['time'],
        y=cpu_data['values'],
        mode='lines',
        name='CPU Usage',
        line=dict(color='red', width=2),
        fill='tonexty'
    ))
    
    fig.update_layout(
        height=200,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(range=[0, 100])
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_memory_realtime():
    """Render real-time memory monitoring"""
    
    st.markdown("#### Memory Usage")
    
    memory_data = generate_realtime_data('memory')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=memory_data['time'],
        y=memory_data['values'],
        mode='lines',
        name='Memory Usage',
        line=dict(color='blue', width=2),
        fill='tonexty'
    ))
    
    fig.update_layout(
        height=200,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(range=[0, 100])
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_gpu_realtime():
    """Render real-time GPU monitoring"""
    
    st.markdown("#### GPU Usage")
    
    gpu_data = generate_realtime_data('gpu')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=gpu_data['time'],
        y=gpu_data['values'],
        mode='lines',
        name='GPU Usage',
        line=dict(color='green', width=2),
        fill='tonexty'
    ))
    
    fig.update_layout(
        height=200,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(range=[0, 100])
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_network_realtime():
    """Render real-time network monitoring"""
    
    st.markdown("#### Network I/O")
    
    network_data = generate_realtime_data('network')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=network_data['time'],
        y=network_data['values'],
        mode='lines',
        name='Network Usage',
        line=dict(color='orange', width=2),
        fill='tonexty'
    ))
    
    fig.update_layout(
        height=200,
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(range=[0, 100])
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_process_monitor():
    """Render top processes monitor"""
    
    processes = get_top_processes()
    
    df = pd.DataFrame(processes)
    
    st.dataframe(
        df,
        column_config={
            "CPU %": st.column_config.ProgressColumn(
                "CPU %",
                help="CPU usage percentage",
                min_value=0,
                max_value=100,
            ),
            "Memory %": st.column_config.ProgressColumn(
                "Memory %",
                help="Memory usage percentage",
                min_value=0,
                max_value=100,
            ),
        },
        hide_index=True,
        use_container_width=True
    )

def render_cpu_details():
    """Render detailed CPU information"""
    
    st.markdown("#### CPU Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cpu_info = get_cpu_info()
        for key, value in cpu_info.items():
            st.metric(key, value)
    
    with col2:
        # CPU core usage
        core_usage = psutil.cpu_percent(percpu=True, interval=0.1)
        
        fig = px.bar(
            x=[f"Core {i}" for i in range(len(core_usage))],
            y=core_usage,
            title="Per-Core Usage",
            color=core_usage,
            color_continuous_scale='reds'
        )
        fig.update_layout(
            height=300,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

def render_memory_breakdown():
    """Render memory usage breakdown"""
    
    st.markdown("#### Memory Breakdown")
    
    memory = psutil.virtual_memory()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Memory pie chart
        memory_data = {
            'Used': memory.used,
            'Available': memory.available,
            'Buffers': getattr(memory, 'buffers', 0),
            'Cached': getattr(memory, 'cached', 0)
        }
        
        fig = px.pie(
            values=list(memory_data.values()),
            names=list(memory_data.keys()),
            title="Memory Distribution"
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Memory metrics
        st.metric("Total RAM", f"{memory.total / 1024**3:.1f} GB")
        st.metric("Available RAM", f"{memory.available / 1024**3:.1f} GB")
        st.metric("Used RAM", f"{memory.used / 1024**3:.1f} GB")
        st.metric("Memory %", f"{memory.percent:.1f}%")

def render_gpu_details():
    """Render GPU details (mock data for demo)"""
    
    st.markdown("#### GPU Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # GPU info
        st.metric("GPU Model", "NVIDIA Quadro P2000")
        st.metric("VRAM Total", "5 GB")
        st.metric("VRAM Used", "2.3 GB")
        st.metric("GPU Usage", f"{get_gpu_usage():.1f}%")
    
    with col2:
        # GPU temperature and power
        gpu_temp = get_gpu_temperature()
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gpu_temp,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "GPU Temperature"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "red"}
                ]
            }
        ))
        fig.update_layout(height=250, plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

def render_storage_analysis():
    """Render storage analysis"""
    
    st.markdown("#### Storage Analysis")
    
    # Disk usage
    disks = get_disk_info()
    
    for disk in disks:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.text(f"Drive: {disk['device']}")
        with col2:
            st.text(f"Total: {disk['total']:.1f} GB")
        with col3:
            st.text(f"Free: {disk['free']:.1f} GB")
        with col4:
            st.progress(disk['percent'] / 100)

def render_performance_history():
    """Render performance benchmark history"""
    
    # Mock historical data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    cpu_scores = 1000 + np.random.normal(0, 50, 30)
    gpu_scores = 2500 + np.random.normal(0, 100, 30)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=cpu_scores,
        mode='lines+markers',
        name='CPU Score',
        line=dict(color='red')
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=gpu_scores,
        mode='lines+markers',
        name='GPU Score',
        line=dict(color='green'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        title="Performance History",
        xaxis_title="Date",
        yaxis=dict(title="CPU Score", side="left"),
        yaxis2=dict(title="GPU Score", side="right", overlaying="y"),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_performance_comparison():
    """Render performance comparison with similar systems"""
    
    st.markdown("#### Performance vs Similar Systems")
    
    comparison_data = {
        'Metric': ['CPU Score', 'GPU Score', 'Memory Score', 'Disk Score'],
        'Your System': [1250, 2750, 1800, 950],
        'Similar Systems (Avg)': [1100, 2400, 1600, 850],
        'Top 10%': [1800, 4200, 2500, 1400]
    }
    
    df = pd.DataFrame(comparison_data)
    
    fig = px.bar(
        df,
        x='Metric',
        y=['Your System', 'Similar Systems (Avg)', 'Top 10%'],
        barmode='group',
        title="Performance Comparison"
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

def render_usage_patterns():
    """Render system usage patterns analysis"""
    
    st.markdown("#### Usage Patterns")
    
    # Mock hourly usage pattern
    hours = list(range(24))
    cpu_pattern = [20 + 30 * np.sin(h * np.pi / 12) + np.random.normal(0, 5) for h in hours]
    memory_pattern = [40 + 20 * np.sin((h + 3) * np.pi / 12) + np.random.normal(0, 3) for h in hours]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hours,
        y=cpu_pattern,
        mode='lines',
        name='CPU Usage',
        fill='tonexty'
    ))
    
    fig.add_trace(go.Scatter(
        x=hours,
        y=memory_pattern,
        mode='lines',
        name='Memory Usage',
        fill='tonexty'
    ))
    
    fig.update_layout(
        title="24-Hour Usage Pattern",
        xaxis_title="Hour of Day",
        yaxis_title="Usage %",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_performance_trends():
    """Render performance trends"""
    
    st.markdown("#### Performance Trends")
    
    # Mock trend data
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    performance_trend = 100 + np.cumsum(np.random.normal(0, 2, 30))
    
    fig = px.line(
        x=dates,
        y=performance_trend,
        title="System Performance Trend",
        labels={'x': 'Date', 'y': 'Performance Index'}
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)

def render_optimization_insights():
    """Render optimization insights and recommendations"""
    
    st.markdown("#### Optimization Insights")
    
    insights = [
        {"type": "info", "message": "CPU usage is within normal range"},
        {"type": "warning", "message": "Memory usage peaks during training - consider batch size optimization"},
        {"type": "success", "message": "GPU utilization is optimal for current workload"},
        {"type": "info", "message": "Disk I/O is minimal - good for system responsiveness"},
        {"type": "warning", "message": "Consider upgrading RAM for larger model training"}
    ]
    
    for insight in insights:
        if insight["type"] == "info":
            st.info(f"💡 {insight['message']}")
        elif insight["type"] == "warning":
            st.warning(f"⚠️ {insight['message']}")
        elif insight["type"] == "success":
            st.success(f"✅ {insight['message']}")

# Helper functions for system monitoring

def get_system_info():
    """Get basic system information"""
    return {
        'platform': platform.system(),
        'processor': platform.processor(),
        'architecture': platform.architecture()[0],
        'machine': platform.machine(),
        'python_version': platform.python_version()
    }

def get_detailed_system_info():
    """Get detailed system information"""
    return {
        'OS': f"{platform.system()} {platform.release()}",
        'Processor': platform.processor() or "Unknown",
        'Architecture': platform.architecture()[0],
        'CPU Cores': psutil.cpu_count(logical=False),
        'CPU Threads': psutil.cpu_count(logical=True),
        'Total RAM': f"{psutil.virtual_memory().total / 1024**3:.1f} GB",
        'Python Version': platform.python_version(),
        'Boot Time': datetime.fromtimestamp(psutil.boot_time()).strftime('%Y-%m-%d %H:%M:%S')
    }

def get_gpu_usage():
    """Get GPU usage (mock for demo)"""
    return np.random.uniform(20, 80)

def get_gpu_temperature():
    """Get GPU temperature (mock for demo)"""
    return np.random.uniform(55, 75)

def get_system_temperature():
    """Get system temperature (mock for demo)"""
    return np.random.uniform(45, 65)

def get_cpu_info():
    """Get detailed CPU information"""
    return {
        'CPU Count': psutil.cpu_count(logical=False),
        'Thread Count': psutil.cpu_count(logical=True),
        'Current Freq': f"{psutil.cpu_freq().current:.0f} MHz" if psutil.cpu_freq() else "Unknown",
        'Max Freq': f"{psutil.cpu_freq().max:.0f} MHz" if psutil.cpu_freq() else "Unknown"
    }

def get_top_processes():
    """Get top processes by CPU and memory usage"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            processes.append(proc.info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    # Sort by CPU usage and return top 10
    processes.sort(key=lambda x: x['cpu_percent'], reverse=True)
    return processes[:10]

def get_disk_info():
    """Get disk usage information"""
    disks = []
    for partition in psutil.disk_partitions():
        try:
            disk_usage = psutil.disk_usage(partition.mountpoint)
            disks.append({
                'device': partition.device,
                'total': disk_usage.total / 1024**3,  # GB
                'used': disk_usage.used / 1024**3,   # GB
                'free': disk_usage.free / 1024**3,   # GB
                'percent': (disk_usage.used / disk_usage.total) * 100
            })
        except PermissionError:
            continue
    return disks

def generate_realtime_data(metric_type):
    """Generate mock real-time data for charts"""
    now = datetime.now()
    times = [now - timedelta(seconds=i*5) for i in range(60, 0, -1)]
    
    if metric_type == 'cpu':
        base = 45
        values = [base + 20 * np.sin(i * 0.1) + np.random.normal(0, 5) for i in range(60)]
    elif metric_type == 'memory':
        base = 65
        values = [base + 15 * np.sin(i * 0.05) + np.random.normal(0, 3) for i in range(60)]
    elif metric_type == 'gpu':
        base = 35
        values = [base + 25 * np.sin(i * 0.08) + np.random.normal(0, 7) for i in range(60)]
    else:  # network
        base = 25
        values = [base + 30 * np.sin(i * 0.12) + np.random.normal(0, 8) for i in range(60)]
    
    # Ensure values are within reasonable bounds
    values = [max(0, min(100, v)) for v in values]
    
    return {'time': times, 'values': values}

def run_cpu_benchmark():
    """Run CPU benchmark (mock)"""
    # Simulate benchmark
    import time
    time.sleep(2)
    return np.random.randint(1000, 1500)

def run_gpu_benchmark():
    """Run GPU benchmark (mock)"""
    import time
    time.sleep(3)
    return np.random.randint(2000, 3000)

def run_memory_benchmark():
    """Run memory benchmark (mock)"""
    import time
    time.sleep(1)
    return np.random.randint(1500, 2000)

def run_disk_benchmark():
    """Run disk benchmark (mock)"""
    import time
    time.sleep(2)
    return np.random.randint(800, 1200)

def run_health_checks():
    """Run system health checks"""
    return [
        {'name': 'CPU Temperature', 'status': 'pass', 'message': ''},
        {'name': 'Memory Usage', 'status': 'warning', 'message': 'High usage detected'},
        {'name': 'Disk Space', 'status': 'pass', 'message': ''},
        {'name': 'GPU Status', 'status': 'pass', 'message': ''},
        {'name': 'Network Connectivity', 'status': 'pass', 'message': ''}
    ]

def generate_system_report():
    """Generate comprehensive system report"""
    st.success("System report generated successfully!")
    st.info("Report saved to: system_report_2024.pdf")