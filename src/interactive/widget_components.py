"""Reusable widget components for interactive interfaces."""

import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import io
import base64
from PIL import Image

from config import ModelConfig, VisualizationConfig
from src.models import TransformerEncoder
from src.positional_encoding import get_positional_encoding
from src.utils.tokenizer import SimpleTokenizer
from src.utils.metrics import AttentionMetrics, EncodingMetrics


class ParameterWidget:
    """Interactive parameter control widget."""
    
    def __init__(self, config: ModelConfig, on_change: Optional[Callable] = None):
        self.config = config
        self.on_change = on_change
        self.widgets = {}
        self.create_widgets()
    
    def create_widgets(self):
        """Create parameter control widgets."""
        # Model architecture parameters
        self.widgets['d_model'] = widgets.Dropdown(
            options=[64, 128, 256, 512, 1024],
            value=self.config.d_model,
            description='Model Dim:',
            style={'description_width': '120px'}
        )
        
        self.widgets['n_heads'] = widgets.Dropdown(
            options=[1, 2, 4, 8, 12, 16],
            value=self.config.n_heads,
            description='Attention Heads:',
            style={'description_width': '120px'}
        )
        
        self.widgets['n_layers'] = widgets.IntSlider(
            min=1, max=12, value=self.config.n_layers,
            description='Layers:',
            style={'description_width': '120px'}
        )
        
        # Positional encoding parameters
        self.widgets['encoding_type'] = widgets.Dropdown(
            options=['sinusoidal', 'learned', 'rope', 'relative'],
            value=self.config.encoding_type,
            description='Encoding:',
            style={'description_width': '120px'}
        )
        
        # Sequence parameters
        self.widgets['max_seq_len'] = widgets.IntSlider(
            min=16, max=1024, step=16, value=self.config.max_seq_len,
            description='Max Seq Len:',
            style={'description_width': '120px'}
        )
        
        # Training parameters
        self.widgets['dropout'] = widgets.FloatSlider(
            min=0.0, max=0.5, step=0.05, value=self.config.dropout,
            description='Dropout:',
            style={'description_width': '120px'}
        )
        
        # Encoding-specific parameters
        self.widgets['rope_theta'] = widgets.FloatLogSlider(
            min=3, max=5, step=0.1, value=np.log10(self.config.rope_theta),
            description='RoPE θ (log):',
            style={'description_width': '120px'},
            layout=widgets.Layout(display='none')
        )
        
        # Connect callbacks
        for name, widget in self.widgets.items():
            widget.observe(self._on_parameter_change, names='value')
        
        # Show/hide encoding-specific parameters
        self.widgets['encoding_type'].observe(self._update_encoding_params, names='value')
    
    def _on_parameter_change(self, change):
        """Handle parameter changes."""
        widget_name = None
        for name, widget in self.widgets.items():
            if widget is change['owner']:
                widget_name = name
                break
        
        if widget_name:
            # Update config
            if widget_name == 'rope_theta':
                value = 10 ** change['new']  # Convert from log scale
            else:
                value = change['new']
            
            setattr(self.config, widget_name, value)
            
            # Call change callback
            if self.on_change:
                self.on_change(widget_name, value)
    
    def _update_encoding_params(self, change):
        """Update visibility of encoding-specific parameters."""
        encoding_type = change['new']
        
        if encoding_type == 'rope':
            self.widgets['rope_theta'].layout.display = 'block'
        else:
            self.widgets['rope_theta'].layout.display = 'none'
    
    def display(self):
        """Display the parameter widgets."""
        # Group widgets
        architecture_group = widgets.VBox([
            widgets.HTML("<h3>🏗️ Architecture</h3>"),
            self.widgets['d_model'],
            self.widgets['n_heads'],
            self.widgets['n_layers'],
        ])
        
        encoding_group = widgets.VBox([
            widgets.HTML("<h3>📍 Positional Encoding</h3>"),
            self.widgets['encoding_type'],
            self.widgets['rope_theta'],
        ])
        
        sequence_group = widgets.VBox([
            widgets.HTML("<h3>📝 Sequence</h3>"),
            self.widgets['max_seq_len'],
        ])
        
        training_group = widgets.VBox([
            widgets.HTML("<h3>🎓 Training</h3>"),
            self.widgets['dropout'],
        ])
        
        # Layout in accordion
        accordion = widgets.Accordion(children=[
            architecture_group,
            encoding_group,
            sequence_group,
            training_group
        ])
        
        accordion.set_title(0, 'Architecture')
        accordion.set_title(1, 'Positional Encoding')
        accordion.set_title(2, 'Sequence Settings')
        accordion.set_title(3, 'Training Settings')
        
        display(accordion)
    
    def get_config(self) -> ModelConfig:
        """Get current configuration."""
        return self.config


class VisualizationWidget:
    """Interactive visualization widget."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.output = widgets.Output()
        self.controls = {}
        self.current_data = None
        self.create_controls()
    
    def create_controls(self):
        """Create visualization control widgets."""
        self.controls['plot_type'] = widgets.Dropdown(
            options=[
                ('Heatmap', 'heatmap'),
                ('Line Plot', 'line'),
                ('3D Surface', '3d'),
                ('Scatter Plot', 'scatter'),
                ('Bar Chart', 'bar')
            ],
            value='heatmap',
            description='Plot Type:',
            style={'description_width': '100px'}
        )
        
        self.controls['colormap'] = widgets.Dropdown(
            options=[
                ('Viridis', 'viridis'),
                ('Plasma', 'plasma'),
                ('Blues', 'blues'),
                ('Reds', 'reds'),
                ('RdBu', 'RdBu')
            ],
            value='viridis',
            description='Colormap:',
            style={'description_width': '100px'}
        )
        
        self.controls['show_values'] = widgets.Checkbox(
            value=True,
            description='Show Values',
            style={'description_width': '100px'}
        )
        
        self.controls['max_dimensions'] = widgets.IntSlider(
            min=8, max=64, step=8, value=32,
            description='Max Dims:',
            style={'description_width': '100px'}
        )
        
        # Connect callbacks
        for control in self.controls.values():
            control.observe(self._update_visualization, names='value')
    
    def _update_visualization(self, change):
        """Update visualization when controls change."""
        if self.current_data is not None:
            self.plot(self.current_data)
    
    def plot(self, data: Any, title: str = "Visualization"):
        """Plot data with current settings."""
        self.current_data = data
        
        with self.output:
            clear_output(wait=True)
            
            plot_type = self.controls['plot_type'].value
            colormap = self.controls['colormap'].value
            
            try:
                if isinstance(data, torch.Tensor):
                    data_np = data.detach().cpu().numpy()
                elif isinstance(data, np.ndarray):
                    data_np = data
                else:
                    display(HTML("<p>❌ Unsupported data type</p>"))
                    return
                
                if plot_type == 'heatmap':
                    self._plot_heatmap(data_np, title, colormap)
                elif plot_type == 'line':
                    self._plot_line(data_np, title)
                elif plot_type == '3d':
                    self._plot_3d_surface(data_np, title, colormap)
                elif plot_type == 'scatter':
                    self._plot_scatter(data_np, title)
                elif plot_type == 'bar':
                    self._plot_bar(data_np, title)
                    
            except Exception as e:
                display(HTML(f"<p>❌ Error creating plot: {str(e)}</p>"))
    
    def _plot_heatmap(self, data, title, colormap):
        """Create heatmap visualization."""
        if data.ndim == 1:
            data = data.reshape(1, -1)
        elif data.ndim > 2:
            data = data.reshape(data.shape[0], -1)
        
        fig = go.Figure(data=go.Heatmap(
            z=data.T if data.shape[0] < data.shape[1] else data,
            colorscale=colormap,
            hovertemplate='X: %{x}<br>Y: %{y}<br>Value: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Position" if data.shape[0] < data.shape[1] else "Sample",
            yaxis_title="Dimension" if data.shape[0] < data.shape[1] else "Feature",
            height=500
        )
        
        fig.show()
    
    def _plot_line(self, data, title):
        """Create line plot visualization."""
        fig = go.Figure()
        
        if data.ndim == 1:
            fig.add_trace(go.Scatter(
                y=data,
                mode='lines+markers',
                name='Signal'
            ))
        elif data.ndim == 2:
            max_lines = min(8, data.shape[1])
            for i in range(max_lines):
                fig.add_trace(go.Scatter(
                    y=data[:, i],
                    mode='lines+markers',
                    name=f'Dim {i}',
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Position",
            yaxis_title="Value",
            height=500
        )
        
        fig.show()
    
    def _plot_3d_surface(self, data, title, colormap):
        """Create 3D surface plot."""
        if data.ndim != 2:
            display(HTML("<p>⚠️ 3D surface requires 2D data</p>"))
            return
        
        # Limit data size for performance
        max_dims = self.controls['max_dimensions'].value
        if data.shape[1] > max_dims:
            data = data[:, :max_dims]
        
        x = np.arange(data.shape[0])
        y = np.arange(data.shape[1])
        X, Y = np.meshgrid(x, y)
        
        fig = go.Figure(data=[go.Surface(
            x=X, y=Y, z=data.T,
            colorscale=colormap,
            hovertemplate='X: %{x}<br>Y: %{y}<br>Z: %{z:.3f}<extra></extra>'
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="Position",
                yaxis_title="Dimension",
                zaxis_title="Value",
                camera=dict(eye=dict(x=1.2, y=1.2, z=1.2))
            ),
            height=600
        )
        
        fig.show()
    
    def _plot_scatter(self, data, title):
        """Create scatter plot visualization."""
        if data.ndim == 1:
            fig = go.Figure(data=go.Scatter(
                x=np.arange(len(data)),
                y=data,
                mode='markers',
                name='Data Points'
            ))
        elif data.ndim == 2 and data.shape[1] >= 2:
            fig = go.Figure(data=go.Scatter(
                x=data[:, 0],
                y=data[:, 1],
                mode='markers',
                name='Data Points',
                text=[f'Point {i}' for i in range(len(data))],
                hovertemplate='X: %{x:.3f}<br>Y: %{y:.3f}<br>%{text}<extra></extra>'
            ))
        else:
            display(HTML("<p>⚠️ Scatter plot requires 1D or 2D+ data</p>"))
            return
        
        fig.update_layout(
            title=title,
            xaxis_title="X" if data.ndim > 1 else "Index",
            yaxis_title="Y" if data.ndim > 1 else "Value",
            height=500
        )
        
        fig.show()
    
    def _plot_bar(self, data, title):
        """Create bar chart visualization."""
        if data.ndim == 1:
            fig = go.Figure(data=go.Bar(
                x=np.arange(len(data)),
                y=data,
                name='Values'
            ))
        elif data.ndim == 2:
            # Plot first few dimensions as grouped bars
            fig = go.Figure()
            max_dims = min(8, data.shape[1])
            
            for i in range(max_dims):
                fig.add_trace(go.Bar(
                    x=np.arange(data.shape[0]),
                    y=data[:, i],
                    name=f'Dim {i}',
                    opacity=0.7
                ))
        else:
            display(HTML("<p>⚠️ Bar chart requires 1D or 2D data</p>"))
            return
        
        fig.update_layout(
            title=title,
            xaxis_title="Position",
            yaxis_title="Value",
            barmode='group' if data.ndim > 1 else None,
            height=500
        )
        
        fig.show()
    
    def display(self):
        """Display the visualization widget."""
        controls_box = widgets.HBox([
            self.controls['plot_type'],
            self.controls['colormap'],
            self.controls['show_values'],
            self.controls['max_dimensions']
        ])
        
        widget_box = widgets.VBox([
            widgets.HTML("<h3>📊 Visualization Controls</h3>"),
            controls_box,
            self.output
        ])
        
        display(widget_box)


class ComparisonWidget:
    """Widget for comparing multiple datasets or models."""
    
    def __init__(self):
        self.output = widgets.Output()
        self.data_storage = {}
        self.controls = {}
        self.create_controls()
    
    def create_controls(self):
        """Create comparison control widgets."""
        self.controls['comparison_type'] = widgets.Dropdown(
            options=[
                ('Side by Side', 'side_by_side'),
                ('Overlay', 'overlay'),
                ('Difference', 'difference'),
                ('Correlation', 'correlation')
            ],
            value='side_by_side',
            description='Comparison:',
            style={'description_width': '100px'}
        )
        
        self.controls['metric'] = widgets.Dropdown(
            options=[
                ('Raw Values', 'raw'),
                ('Normalized', 'normalized'),
                ('Z-Score', 'zscore'),
                ('Percentile', 'percentile')
            ],
            value='raw',
            description='Metric:',
            style={'description_width': '100px'}
        )
        
        self.controls['update_btn'] = widgets.Button(
            description='🔄 Update Comparison',
            button_style='primary'
        )
        
        # Connect callbacks
        self.controls['update_btn'].on_click(self._update_comparison)
    
    def add_data(self, name: str, data: Any, label: str = None):
        """Add data for comparison."""
        if label is None:
            label = name
        
        self.data_storage[name] = {
            'data': data,
            'label': label
        }
    
    def _update_comparison(self, button):
        """Update comparison visualization."""
        if len(self.data_storage) < 2:
            with self.output:
                clear_output(wait=True)
                display(HTML("<p>⚠️ Need at least 2 datasets for comparison</p>"))
            return
        
        comparison_type = self.controls['comparison_type'].value
        metric = self.controls['metric'].value
        
        with self.output:
            clear_output(wait=True)
            
            try:
                if comparison_type == 'side_by_side':
                    self._plot_side_by_side(metric)
                elif comparison_type == 'overlay':
                    self._plot_overlay(metric)
                elif comparison_type == 'difference':
                    self._plot_difference(metric)
                elif comparison_type == 'correlation':
                    self._plot_correlation(metric)
            except Exception as e:
                display(HTML(f"<p>❌ Error in comparison: {str(e)}</p>"))
    
    def _normalize_data(self, data, metric):
        """Normalize data according to metric."""
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        if metric == 'normalized':
            return (data - data.min()) / (data.max() - data.min() + 1e-8)
        elif metric == 'zscore':
            return (data - data.mean()) / (data.std() + 1e-8)
        elif metric == 'percentile':
            return np.percentile(data, np.linspace(0, 100, data.size)).reshape(data.shape)
        else:  # raw
            return data
    
    def _plot_side_by_side(self, metric):
        """Create side-by-side comparison."""
        n_datasets = len(self.data_storage)
        cols = min(3, n_datasets)
        rows = (n_datasets + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[item['label'] for item in self.data_storage.values()]
        )
        
        for i, (name, item) in enumerate(self.data_storage.items()):
            row = i // cols + 1
            col = i % cols + 1
            
            data = self._normalize_data(item['data'], metric)
            
            if data.ndim == 1:
                fig.add_trace(
                    go.Scatter(y=data, name=item['label'], showlegend=False),
                    row=row, col=col
                )
            elif data.ndim == 2:
                fig.add_trace(
                    go.Heatmap(z=data.T, showscale=(i==0)),
                    row=row, col=col
                )
        
        fig.update_layout(
            title="Side-by-Side Comparison",
            height=400 * rows
        )
        
        fig.show()
    
    def _plot_overlay(self, metric):
        """Create overlay comparison."""
        fig = go.Figure()
        
        for name, item in self.data_storage.items():
            data = self._normalize_data(item['data'], metric)
            
            if data.ndim == 1:
                fig.add_trace(go.Scatter(
                    y=data,
                    mode='lines+markers',
                    name=item['label'],
                    line=dict(width=2),
                    opacity=0.8
                ))
            elif data.ndim == 2:
                # Plot first dimension or mean
                if data.shape[1] > 1:
                    y_data = data.mean(axis=1)
                else:
                    y_data = data.flatten()
                
                fig.add_trace(go.Scatter(
                    y=y_data,
                    mode='lines+markers',
                    name=item['label'],
                    line=dict(width=2),
                    opacity=0.8
                ))
        
        fig.update_layout(
            title="Overlay Comparison",
            xaxis_title="Position",
            yaxis_title="Value",
            height=500
        )
        
        fig.show()
    
    def _plot_difference(self, metric):
        """Create difference comparison."""
        data_items = list(self.data_storage.items())
        
        if len(data_items) < 2:
            display(HTML("<p>⚠️ Need at least 2 datasets for difference</p>"))
            return
        
        # Compare first two datasets
        data1 = self._normalize_data(data_items[0][1]['data'], metric)
        data2 = self._normalize_data(data_items[1][1]['data'], metric)
        
        # Ensure same shape
        min_shape = tuple(min(s1, s2) for s1, s2 in zip(data1.shape, data2.shape))
        data1 = data1[:min_shape[0]]
        data2 = data2[:min_shape[0]]
        
        if len(min_shape) > 1:
            data1 = data1[:, :min_shape[1]]
            data2 = data2[:, :min_shape[1]]
        
        difference = data1 - data2
        
        if difference.ndim == 1:
            fig = go.Figure(data=go.Scatter(
                y=difference,
                mode='lines+markers',
                name='Difference',
                line=dict(color='red', width=2)
            ))
        else:
            fig = go.Figure(data=go.Heatmap(
                z=difference.T,
                colorscale='RdBu',
                zmid=0
            ))
        
        fig.update_layout(
            title=f"Difference: {data_items[0][1]['label']} - {data_items[1][1]['label']}",
            xaxis_title="Position",
            yaxis_title="Dimension" if difference.ndim > 1 else "Difference",
            height=500
        )
        
        fig.show()
        
        # Show statistics
        stats_html = f"""
        <div style='background-color: #f0f9ff; padding: 15px; border-radius: 8px; margin-top: 10px;'>
            <h4>📊 Difference Statistics</h4>
            <p><strong>Mean Difference:</strong> {difference.mean():.6f}</p>
            <p><strong>Std Difference:</strong> {difference.std():.6f}</p>
            <p><strong>Max Abs Difference:</strong> {np.abs(difference).max():.6f}</p>
            <p><strong>L2 Norm:</strong> {np.linalg.norm(difference):.6f}</p>
        </div>
        """
        display(HTML(stats_html))
    
    def _plot_correlation(self, metric):
        """Create correlation comparison."""
        data_items = list(self.data_storage.items())
        
        if len(data_items) < 2:
            display(HTML("<p>⚠️ Need at least 2 datasets for correlation</p>"))
            return
        
        # Create correlation matrix between all datasets
        n_datasets = len(data_items)
        correlations = np.zeros((n_datasets, n_datasets))
        labels = []
        
        normalized_data = []
        for name, item in data_items:
            data = self._normalize_data(item['data'], metric)
            normalized_data.append(data.flatten())
            labels.append(item['label'])
        
        # Compute correlations
        for i in range(n_datasets):
            for j in range(n_datasets):
                if i == j:
                    correlations[i, j] = 1.0
                else:
                    # Ensure same length
                    min_len = min(len(normalized_data[i]), len(normalized_data[j]))
                    corr = np.corrcoef(
                        normalized_data[i][:min_len], 
                        normalized_data[j][:min_len]
                    )[0, 1]
                    correlations[i, j] = corr
        
        fig = go.Figure(data=go.Heatmap(
            z=correlations,
            x=labels,
            y=labels,
            colorscale='RdBu',
            zmid=0,
            zmin=-1,
            zmax=1,
            hovertemplate='%{x} vs %{y}<br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Correlation Matrix",
            height=500
        )
        
        fig.show()
    
    def display(self):
        """Display the comparison widget."""
        controls_box = widgets.HBox([
            self.controls['comparison_type'],
            self.controls['metric'],
            self.controls['update_btn']
        ])
        
        widget_box = widgets.VBox([
            widgets.HTML("<h3>⚖️ Comparison Controls</h3>"),
            controls_box,
            widgets.HTML(f"<p>📊 Datasets loaded: {len(self.data_storage)}</p>"),
            self.output
        ])
        
        display(widget_box)
    
    def clear_data(self):
        """Clear all stored data."""
        self.data_storage.clear()
        with self.output:
            clear_output()


class ExportWidget:
    """Widget for exporting visualizations and data."""
    
    def __init__(self):
        self.output = widgets.Output()
        self.controls = {}
        self.export_data = {}
        self.create_controls()
    
    def create_controls(self):
        """Create export control widgets."""
        self.controls['export_format'] = widgets.Dropdown(
            options=[
                ('PNG Image', 'png'),
                ('PDF Document', 'pdf'),
                ('SVG Vector', 'svg'),
                ('HTML Interactive', 'html'),
                ('CSV Data', 'csv'),
                ('JSON Data', 'json'),
                ('Excel Workbook', 'xlsx')
            ],
            value='png',
            description='Format:',
            style={'description_width': '100px'}
        )
        
        self.controls['export_quality'] = widgets.Dropdown(
            options=[
                ('Low (150 DPI)', 150),
                ('Medium (300 DPI)', 300),
                ('High (600 DPI)', 600),
                ('Ultra (1200 DPI)', 1200)
            ],
            value=300,
            description='Quality:',
            style={'description_width': '100px'}
        )
        
        self.controls['include_data'] = widgets.Checkbox(
            value=True,
            description='Include Raw Data',
            style={'description_width': '140px'}
        )
        
        self.controls['include_config'] = widgets.Checkbox(
            value=True,
            description='Include Configuration',
            style={'description_width': '140px'}
        )
        
        self.controls['filename'] = widgets.Text(
            value='positional_encoding_analysis',
            description='Filename:',
            style={'description_width': '100px'}
        )
        
        self.controls['export_btn'] = widgets.Button(
            description='📥 Export',
            button_style='primary',
            icon='download'
        )
        
        # Connect callbacks
        self.controls['export_btn'].on_click(self._perform_export)
    
    def add_export_item(self, name: str, data: Any, metadata: Dict = None):
        """Add item for export."""
        self.export_data[name] = {
            'data': data,
            'metadata': metadata or {},
            'timestamp': pd.Timestamp.now()
        }
    
    def _perform_export(self, button):
        """Perform the export operation."""
        if not self.export_data:
            with self.output:
                clear_output(wait=True)
                display(HTML("<p>⚠️ No data to export</p>"))
            return
        
        export_format = self.controls['export_format'].value
        filename = self.controls['filename'].value
        include_data = self.controls['include_data'].value
        include_config = self.controls['include_config'].value
        quality = self.controls['export_quality'].value
        
        with self.output:
            clear_output(wait=True)
            
            try:
                if export_format in ['png', 'pdf', 'svg']:
                    self._export_visualization(export_format, filename, quality)
                elif export_format == 'html':
                    self._export_html(filename, include_data, include_config)
                elif export_format in ['csv', 'json', 'xlsx']:
                    self._export_data(export_format, filename, include_config)
                
                display(HTML(f"<p>✅ Export completed: {filename}.{export_format}</p>"))
                
            except Exception as e:
                display(HTML(f"<p>❌ Export failed: {str(e)}</p>"))
    
    def _export_visualization(self, format_type, filename, quality):
        """Export visualization files."""
        # This would integrate with the actual plotting libraries
        # For now, create a placeholder message
        display(HTML(f"""
        <div style='background-color: #e3f2fd; padding: 15px; border-radius: 8px;'>
            <h4>📊 Visualization Export</h4>
            <p><strong>Format:</strong> {format_type.upper()}</p>
            <p><strong>Quality:</strong> {quality} DPI</p>
            <p><strong>Filename:</strong> {filename}.{format_type}</p>
            <p><em>Note: In a full implementation, this would save the current visualization.</em></p>
        </div>
        """))
    
    def _export_html(self, filename, include_data, include_config):
        """Export HTML report."""
        html_content = self._generate_html_report(include_data, include_config)
        
        # In a real implementation, this would save to file
        display(HTML(f"""
        <div style='background-color: #f0f9ff; padding: 15px; border-radius: 8px;'>
            <h4>📄 HTML Report Export</h4>
            <p><strong>Filename:</strong> {filename}.html</p>
            <p><strong>Include Data:</strong> {'Yes' if include_data else 'No'}</p>
            <p><strong>Include Config:</strong> {'Yes' if include_config else 'No'}</p>
            <details>
                <summary>Preview HTML Content</summary>
                <div style='max-height: 200px; overflow-y: auto; background: white; padding: 10px; margin-top: 10px;'>
                    {html_content[:500]}...
                </div>
            </details>
        </div>
        """))
    
    def _export_data(self, format_type, filename, include_config):
        """Export data files."""
        # Create summary of export data
        data_summary = []
        for name, item in self.export_data.items():
            data = item['data']
            if isinstance(data, torch.Tensor):
                shape = tuple(data.shape)
                dtype = str(data.dtype)
            elif isinstance(data, np.ndarray):
                shape = data.shape
                dtype = str(data.dtype)
            else:
                shape = "Unknown"
                dtype = type(data).__name__
            
            data_summary.append({
                'Name': name,
                'Shape': str(shape),
                'Type': dtype,
                'Timestamp': item['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            })
        
        summary_df = pd.DataFrame(data_summary)
        
        display(HTML(f"""
        <div style='background-color: #fff3e0; padding: 15px; border-radius: 8px;'>
            <h4>💾 Data Export</h4>
            <p><strong>Format:</strong> {format_type.upper()}</p>
            <p><strong>Filename:</strong> {filename}.{format_type}</p>
            <p><strong>Items:</strong> {len(self.export_data)}</p>
            <p><strong>Include Config:</strong> {'Yes' if include_config else 'No'}</p>
        </div>
        """))
        
        # Display data summary
        display(HTML("<h5>📋 Data Summary:</h5>"))
        display(summary_df)
    
    def _generate_html_report(self, include_data, include_config):
        """Generate HTML report content."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Positional Encoding Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; padding: 20px; border-radius: 8px; }}
                .section {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; 
                          background: white; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🧠 Positional Encoding Analysis Report</h1>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>📊 Summary</h2>
                <p>This report contains analysis of {len(self.export_data)} datasets/visualizations.</p>
        """
        
        if include_config:
            html += """
                <div class="metric">
                    <strong>Configuration Included:</strong> ✅
                </div>
            """
        
        if include_data:
            html += """
                <div class="metric">
                    <strong>Raw Data Included:</strong> ✅
                </div>
            """
        
        html += """
            </div>
            
            <div class="section">
                <h2>📈 Analysis Results</h2>
                <p>Detailed analysis results would be included here in a full implementation.</p>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def display(self):
        """Display the export widget."""
        controls_row1 = widgets.HBox([
            self.controls['export_format'],
            self.controls['export_quality']
        ])
        
        controls_row2 = widgets.HBox([
            self.controls['include_data'],
            self.controls['include_config']
        ])
        
        controls_row3 = widgets.HBox([
            self.controls['filename'],
            self.controls['export_btn']
        ])
        
        widget_box = widgets.VBox([
            widgets.HTML("<h3>📥 Export Controls</h3>"),
            controls_row1,
            controls_row2,
            controls_row3,
            widgets.HTML(f"<p>📦 Items ready for export: {len(self.export_data)}</p>"),
            self.output
        ])
        
        display(widget_box)
    
    def clear_export_data(self):
        """Clear all export data."""
        self.export_data.clear()
        with self.output:
            clear_output()


class InteractiveExplorer:
    """Main interactive explorer combining all widgets."""
    
    def __init__(self, config: ModelConfig = None):
        self.config = config or ModelConfig()
        self.tokenizer = SimpleTokenizer()
        
        # Initialize widgets
        self.parameter_widget = ParameterWidget(self.config, self._on_parameter_change)
        self.visualization_widget = VisualizationWidget(VisualizationConfig())
        self.comparison_widget = ComparisonWidget()
        self.export_widget = ExportWidget()
        
        # Model cache
        self.model_cache = {}
        self.current_encoding = None
        self.current_model = None
        
        # Main output area
        self.main_output = widgets.Output()
        
        # Create main interface
        self.create_interface()
    
    def create_interface(self):
        """Create the main interactive interface."""
        # Tab structure
        self.tabs = widgets.Tab()
        
        # Create tab contents
        encoding_tab = self._create_encoding_tab()
        attention_tab = self._create_attention_tab()
        comparison_tab = self._create_comparison_tab()
        export_tab = self._create_export_tab()
        
        # Add tabs
        self.tabs.children = [encoding_tab, attention_tab, comparison_tab, export_tab]
        self.tabs.set_title(0, '📊 Encoding Analysis')
        self.tabs.set_title(1, '🔍 Attention Patterns')
        self.tabs.set_title(2, '⚖️ Comparison')
        self.tabs.set_title(3, '📥 Export')
    
    def _create_encoding_tab(self):
        """Create encoding analysis tab."""
        # Controls
        seq_len_slider = widgets.IntSlider(
            min=8, max=128, step=8, value=32,
            description='Seq Length:',
            style={'description_width': '100px'}
        )
        
        analyze_btn = widgets.Button(
            description='🔍 Analyze Encoding',
            button_style='primary'
        )
        
        # Output area
        encoding_output = widgets.Output()
        
        def analyze_encoding(button):
            with encoding_output:
                clear_output(wait=True)
                try:
                    # Create encoding
                    encoding = get_positional_encoding(self.config)
                    self.current_encoding = encoding
                    
                    # Get encoding matrix
                    if hasattr(encoding, 'forward'):
                        enc_output = encoding.forward(seq_len_slider.value, self.config.d_model)
                        matrix = enc_output.squeeze(0)
                        
                        # Visualize
                        self.visualization_widget.plot(matrix, 
                                                     f"{self.config.encoding_type.title()} Encoding")
                        
                        # Add to export
                        self.export_widget.add_export_item(
                            f"encoding_{self.config.encoding_type}",
                            matrix,
                            {'seq_len': seq_len_slider.value, 'encoding_type': self.config.encoding_type}
                        )
                        
                        # Add to comparison
                        self.comparison_widget.add_data(
                            f"{self.config.encoding_type}_{seq_len_slider.value}",
                            matrix,
                            f"{self.config.encoding_type.title()} (L={seq_len_slider.value})"
                        )
                        
                        display(HTML("<p>✅ Encoding analysis completed!</p>"))
                    else:
                        display(HTML("<p>❌ Could not generate encoding</p>"))
                        
                except Exception as e:
                    display(HTML(f"<p>❌ Error: {str(e)}</p>"))
        
        analyze_btn.on_click(analyze_encoding)
        
        controls = widgets.VBox([
            widgets.HTML("<h3>🎛️ Encoding Controls</h3>"),
            seq_len_slider,
            analyze_btn
        ])
        
        # Layout
        tab_content = widgets.HBox([
            widgets.VBox([controls, self.parameter_widget.display()]),
            widgets.VBox([encoding_output, self.visualization_widget.output])
        ])
        
        return tab_content
    
    def _create_attention_tab(self):
        """Create attention analysis tab."""
        # Text input
        text_input = widgets.Textarea(
            value="The quick brown fox jumps over the lazy dog.",
            description='Input Text:',
            layout=widgets.Layout(width='100%', height='80px')
        )
        
        # Analysis controls
        layer_slider = widgets.IntSlider(
            min=0, max=self.config.n_layers-1, value=0,
            description='Layer:',
            style={'description_width': '80px'}
        )
        
        head_slider = widgets.IntSlider(
            min=0, max=self.config.n_heads-1, value=0,
            description='Head:',
            style={'description_width': '80px'}
        )
        
        analyze_attention_btn = widgets.Button(
            description='🔍 Analyze Attention',
            button_style='primary'
        )
        
        attention_output = widgets.Output()
        
        def analyze_attention(button):
            with attention_output:
                clear_output(wait=True)
                try:
                    # Tokenize input
                    tokens = self.tokenizer.tokenize(text_input.value)[:32]
                    
                    if len(tokens) < 2:
                        display(HTML("<p>⚠️ Need at least 2 tokens</p>"))
                        return
                    
                    # Get or create model
                    model = self._get_or_create_model()
                    
                    # Compute attention
                    input_ids = torch.tensor([list(range(len(tokens)))])
                    
                    with torch.no_grad():
                        outputs = model.forward(input_ids, store_visualizations=True)
                        attention_weights = outputs['attention_weights']
                    
                    if attention_weights and layer_slider.value < len(attention_weights):
                        layer_attention = attention_weights[layer_slider.value]
                        
                        if layer_attention.dim() == 4:
                            attn_matrix = layer_attention[0, head_slider.value]
                        else:
                            attn_matrix = layer_attention[head_slider.value]
                        
                        # Visualize attention
                        self.visualization_widget.plot(
                            attn_matrix,
                            f"Attention - L{layer_slider.value} H{head_slider.value}"
                        )
                        
                        # Add to export and comparison
                        self.export_widget.add_export_item(
                            f"attention_L{layer_slider.value}_H{head_slider.value}",
                            attn_matrix,
                            {'tokens': tokens, 'layer': layer_slider.value, 'head': head_slider.value}
                        )
                        
                        display(HTML("<p>✅ Attention analysis completed!</p>"))
                        
                        # Show token information
                        tokens_html = "<p><strong>Tokens:</strong> " + " → ".join(tokens) + "</p>"
                        display(HTML(tokens_html))
                    else:
                        display(HTML("<p>❌ No attention data available</p>"))
                        
                except Exception as e:
                    display(HTML(f"<p>❌ Error: {str(e)}</p>"))
        
        analyze_attention_btn.on_click(analyze_attention)
        
        controls = widgets.VBox([
            widgets.HTML("<h3>🎛️ Attention Controls</h3>"),
            text_input,
            widgets.HBox([layer_slider, head_slider]),
            analyze_attention_btn
        ])
        
        tab_content = widgets.HBox([
            controls,
            widgets.VBox([attention_output, self.visualization_widget.output])
        ])
        
        return tab_content
    
    def _create_comparison_tab(self):
        """Create comparison tab."""
        return self.comparison_widget.display()
    
    def _create_export_tab(self):
        """Create export tab."""
        return self.export_widget.display()
    
    def _on_parameter_change(self, param_name, value):
        """Handle parameter changes."""
        # Update sliders when model parameters change
        if param_name == 'n_layers':
            # Update layer slider max value in attention tab
            pass
        elif param_name == 'n_heads':
            # Update head slider max value in attention tab
            pass
        
        # Clear model cache when architecture changes
        if param_name in ['d_model', 'n_heads', 'n_layers', 'encoding_type']:
            self.model_cache.clear()
    
    def _get_or_create_model(self):
        """Get or create model from cache."""
        config_key = f"{self.config.d_model}_{self.config.n_heads}_{self.config.n_layers}_{self.config.encoding_type}"
        
        if config_key not in self.model_cache:
            model = TransformerEncoder(self.config)
            self.model_cache[config_key] = model
        
        return self.model_cache[config_key]
    
    def display(self):
        """Display the complete interactive explorer."""
        # Header
        header = widgets.HTML("""
        <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; border-radius: 10px; margin-bottom: 20px;'>
            <h1>🧠 Interactive Positional Encoding Explorer</h1>
            <p>Explore transformer positional encodings through interactive widgets</p>
        </div>
        """)
        
        # Main interface
        main_interface = widgets.VBox([
            header,
            self.tabs,
            self.main_output
        ])
        
        display(main_interface)


# Convenience functions for creating widget components
def create_parameter_controls(config: ModelConfig = None, on_change: Callable = None) -> ParameterWidget:
    """Create parameter control widget.
    
    Args:
        config: Model configuration
        on_change: Callback function for parameter changes
        
    Returns:
        Parameter widget instance
    """
    if config is None:
        config = ModelConfig()
    
    return ParameterWidget(config, on_change)


def create_visualization_panel(config: VisualizationConfig = None) -> VisualizationWidget:
    """Create visualization panel widget.
    
    Args:
        config: Visualization configuration
        
    Returns:
        Visualization widget instance
    """
    if config is None:
        config = VisualizationConfig()
    
    return VisualizationWidget(config)


def create_comparison_panel() -> ComparisonWidget:
    """Create comparison panel widget.
    
    Returns:
        Comparison widget instance
    """
    return ComparisonWidget()


def create_export_panel() -> ExportWidget:
    """Create export panel widget.
    
    Returns:
        Export widget instance
    """
    return ExportWidget()


def create_interactive_explorer(config: ModelConfig = None) -> InteractiveExplorer:
    """Create complete interactive explorer.
    
    Args:
        config: Model configuration
        
    Returns:
        Interactive explorer instance
    """
    return InteractiveExplorer(config)


# Example usage for Jupyter notebooks
def launch_notebook_interface(config: ModelConfig = None):
    """Launch interactive interface in Jupyter notebook.
    
    Args:
        config: Model configuration
    """
    explorer = create_interactive_explorer(config)
    explorer.display()
    
    return explorer


# Demo function
def demo_widgets():
    """Demonstrate widget functionality."""
    display(HTML("""
    <div style='background-color: #e3f2fd; padding: 20px; border-radius: 10px; margin: 20px 0;'>
        <h2>🎯 Widget Components Demo</h2>
        <p>This demonstrates the various interactive widgets available for positional encoding exploration.</p>
    </div>
    """))
    
    # Create demo configuration
    config = ModelConfig(d_model=128, n_heads=8, encoding_type="sinusoidal")
    
    # Demo parameter widget
    display(HTML("<h3>🎛️ Parameter Controls</h3>"))
    param_widget = create_parameter_controls(config)
    param_widget.display()
    
    # Demo visualization widget
    display(HTML("<h3>📊 Visualization Panel</h3>"))
    viz_widget = create_visualization_panel()
    
    # Create demo data
    demo_data = torch.randn(32, 64)  # 32 positions, 64 dimensions
    viz_widget.plot(demo_data, "Demo Encoding Pattern")
    viz_widget.display()
    
    # Demo comparison widget
    display(HTML("<h3>⚖️ Comparison Panel</h3>"))
    comp_widget = create_comparison_panel()
    
    # Add demo data for comparison
    comp_widget.add_data("sinusoidal", torch.sin(torch.linspace(0, 4*np.pi, 100)), "Sinusoidal")
    comp_widget.add_data("cosine", torch.cos(torch.linspace(0, 4*np.pi, 100)), "Cosine")
    comp_widget.display()
    
    # Demo export widget
    display(HTML("<h3>📥 Export Panel</h3>"))
    export_widget = create_export_panel()
    export_widget.add_export_item("demo_data", demo_data, {"description": "Demo encoding data"})
    export_widget.display()
    
    return {
        'parameter_widget': param_widget,
        'visualization_widget': viz_widget,
        'comparison_widget': comp_widget,
        'export_widget': export_widget
    }


if __name__ == "__main__":
    # Run demo if executed directly
    demo_widgets()

