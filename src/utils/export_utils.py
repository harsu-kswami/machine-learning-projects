"""Export utilities for visualizations, data, and analysis reports."""

import torch
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import numpy as np
import json
import pickle
import csv
import h5py
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime
import base64
import io
from PIL import Image

from config import VisualizationConfig, ModelConfig


class FigureExporter:
    """Export matplotlib and plotly figures in various formats."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        
        # Supported formats
        self.matplotlib_formats = ['png', 'pdf', 'svg', 'eps', 'jpg']
        self.plotly_formats = ['png', 'pdf', 'svg', 'html', 'json']
    
    def export_matplotlib_figure(
        self,
        fig: plt.Figure,
        filepath: str,
        format: Optional[str] = None,
        dpi: Optional[int] = None,
        bbox_inches: str = 'tight',
        transparent: Optional[bool] = None,
        **kwargs
    ) -> str:
        """Export matplotlib figure.
        
        Args:
            fig: Matplotlib figure
            filepath: Output file path
            format: Export format (inferred from filepath if None)
            dpi: DPI for raster formats
            bbox_inches: Bounding box setting
            transparent: Transparent background
            **kwargs: Additional arguments for savefig
            
        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        
        if format is None:
            format = filepath.suffix.lower().lstrip('.')
        
        if format not in self.matplotlib_formats:
            raise ValueError(f"Unsupported format: {format}. Supported: {self.matplotlib_formats}")
        
        # Use config defaults if not specified
        if dpi is None:
            dpi = self.config.export_quality
        if transparent is None:
            transparent = self.config.export_transparent
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save figure
        fig.savefig(
            filepath,
            format=format,
            dpi=dpi,
            bbox_inches=bbox_inches,
            transparent=transparent,
            **kwargs
        )
        
        return str(filepath)
    
    def export_plotly_figure(
        self,
        fig: go.Figure,
        filepath: str,
        format: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        **kwargs
    ) -> str:
        """Export plotly figure.
        
        Args:
            fig: Plotly figure
            filepath: Output file path
            format: Export format
            width: Image width for raster formats
            height: Image height for raster formats
            **kwargs: Additional export arguments
            
        Returns:
            Path to saved file
        """
        filepath = Path(filepath)
        
        if format is None:
            format = filepath.suffix.lower().lstrip('.')
        
        if format not in self.plotly_formats:
            raise ValueError(f"Unsupported format: {format}. Supported: {self.plotly_formats}")
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Export based on format
        if format == 'html':
            fig.write_html(str(filepath), **kwargs)
        elif format == 'json':
            fig.write_json(str(filepath), **kwargs)
        elif format in ['png', 'pdf', 'svg']:
            # Use default dimensions if not specified
            if width is None:
                width = self.config.figure_size[0] * 100
            if height is None:
                height = self.config.figure_size[1] * 100
            
            if format == 'png':
                fig.write_image(str(filepath), format='png', width=width, height=height, **kwargs)
            elif format == 'pdf':
                fig.write_image(str(filepath), format='pdf', width=width, height=height, **kwargs)
            elif format == 'svg':
                fig.write_image(str(filepath), format='svg', width=width, height=height, **kwargs)
        
        return str(filepath)
    
    def export_multiple_figures(
        self,
        figures: Dict[str, Union[plt.Figure, go.Figure]],
        output_dir: str,
        format: str = 'png',
        prefix: str = '',
        suffix: str = ''
    ) -> List[str]:
        """Export multiple figures at once.
        
        Args:
            figures: Dictionary of figure name to figure object
            output_dir: Output directory
            format: Export format
            prefix: Filename prefix
            suffix: Filename suffix
            
        Returns:
            List of saved file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for name, fig in figures.items():
            filename = f"{prefix}{name}{suffix}.{format}"
            filepath = output_dir / filename
            
            if isinstance(fig, plt.Figure):
                saved_path = self.export_matplotlib_figure(fig, filepath, format)
            elif isinstance(fig, go.Figure):
                saved_path = self.export_plotly_figure(fig, filepath, format)
            else:
                print(f"Warning: Unknown figure type for {name}, skipping")
                continue
            
            saved_paths.append(saved_path)
        
        return saved_paths
    
    def create_pdf_report(
        self,
        figures: Dict[str, plt.Figure],
        output_path: str,
        title: str = "Analysis Report",
        metadata: Optional[Dict[str, str]] = None
    ) -> str:
        """Create PDF report with multiple figures.
        
        Args:
            figures: Dictionary of figure name to matplotlib figure
            output_path: Output PDF path
            title: Report title
            metadata: PDF metadata
            
        Returns:
            Path to created PDF
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with matplotlib.backends.backend_pdf.PdfPages(output_path) as pdf:
            # Set metadata
            d = pdf.infodict()
            d['Title'] = title
            d['Author'] = 'Positional Encoding Visualizer'
            d['Subject'] = 'Transformer Analysis Report'
            d['Creator'] = 'Python/Matplotlib'
            d['CreationDate'] = datetime.now()
            
            if metadata:
                d.update(metadata)
            
            # Add figures to PDF
            for name, fig in figures.items():
                # Add title to figure if not present
                if not fig._suptitle:
                    fig.suptitle(name, fontsize=16, y=0.98)
                
                pdf.savefig(fig, bbox_inches='tight')
        
        return str(output_path)


class DataExporter:
    """Export analysis data in various formats."""
    
    def __init__(self):
        self.supported_formats = ['csv', 'json', 'pickle', 'hdf5', 'xlsx']
    
    def export_attention_weights(
        self,
        attention_weights: List[torch.Tensor],
        output_path: str,
        tokens: Optional[List[str]] = None,
        layer_names: Optional[List[str]] = None,
        format: str = 'hdf5'
    ) -> str:
        """Export attention weights to file.
        
        Args:
            attention_weights: List of attention weight tensors
            output_path: Output file path
            tokens: Token strings
            layer_names: Layer names
            format: Export format
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'hdf5':
            with h5py.File(output_path, 'w') as f:
                f.attrs['export_date'] = datetime.now().isoformat()
                f.attrs['num_layers'] = len(attention_weights)
                
                if tokens:
                    f.create_dataset('tokens', data=[t.encode('utf-8') for t in tokens])
                
                for i, weights in enumerate(attention_weights):
                    layer_name = layer_names[i] if layer_names else f'layer_{i}'
                    weights_np = weights.detach().cpu().numpy()
                    f.create_dataset(f'attention/{layer_name}', data=weights_np)
                    f[f'attention/{layer_name}'].attrs['shape'] = weights_np.shape
                    f[f'attention/{layer_name}'].attrs['layer_index'] = i
        
        elif format == 'pickle':
            data = {
                'attention_weights': [w.detach().cpu() for w in attention_weights],
                'tokens': tokens,
                'layer_names': layer_names,
                'export_date': datetime.now().isoformat()
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
        
        elif format == 'json':
            # Convert tensors to lists for JSON serialization
            data = {
                'attention_weights': [w.detach().cpu().numpy().tolist() for w in attention_weights],
                'tokens': tokens,
                'layer_names': layer_names,
                'export_date': datetime.now().isoformat()
            }
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return str(output_path)
    
    def export_encoding_matrices(
        self,
        encoding_matrices: Dict[str, torch.Tensor],
        output_path: str,
        format: str = 'hdf5',
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Export positional encoding matrices.
        
        Args:
            encoding_matrices: Dictionary of encoding name to matrix
            output_path: Output file path
            format: Export format
            metadata: Additional metadata
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'hdf5':
            with h5py.File(output_path, 'w') as f:
                f.attrs['export_date'] = datetime.now().isoformat()
                f.attrs['num_encodings'] = len(encoding_matrices)
                
                if metadata:
                    for key, value in metadata.items():
                        f.attrs[key] = value
                
                for enc_name, matrix in encoding_matrices.items():
                    matrix_np = matrix.detach().cpu().numpy()
                    f.create_dataset(f'encodings/{enc_name}', data=matrix_np)
                    f[f'encodings/{enc_name}'].attrs['shape'] = matrix_np.shape
                    f[f'encodings/{enc_name}'].attrs['encoding_type'] = enc_name
        
        elif format == 'pickle':
            data = {
                'encoding_matrices': {name: matrix.detach().cpu() for name, matrix in encoding_matrices.items()},
                'metadata': metadata,
                'export_date': datetime.now().isoformat()
            }
            
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
        
        return str(output_path)
    
    def export_metrics(
        self,
        metrics: Dict[str, Any],
        output_path: str,
        format: str = 'json'
    ) -> str:
        """Export analysis metrics.
        
        Args:
            metrics: Metrics dictionary
            output_path: Output file path
            format: Export format
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert tensors to lists/floats for serialization
        serializable_metrics = self._make_serializable(metrics)
        serializable_metrics['export_date'] = datetime.now().isoformat()
        
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(serializable_metrics, f, indent=2)
        
        elif format == 'csv':
            # Flatten metrics for CSV export
            flattened = self._flatten_dict(serializable_metrics)
            df = pd.DataFrame([flattened])
            df.to_csv(output_path, index=False)
        
        elif format == 'xlsx':
            # Create Excel file with multiple sheets if needed
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Main metrics sheet
                main_df = pd.DataFrame([self._flatten_dict(serializable_metrics)])
                main_df.to_sheet(writer, sheet_name='Metrics', index=False)
                
                # Additional sheets for complex data
                for key, value in serializable_metrics.items():
                    if isinstance(value, list) and len(value) > 0:
                        if isinstance(value[0], dict):
                            df = pd.DataFrame(value)
                            df.to_excel(writer, sheet_name=key[:31], index=False)
        
        return str(output_path)
    
    def _make_serializable(self, obj: Any) -> Any:
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def export_dataset(
        self,
        dataset: Dict[str, torch.Tensor],
        output_path: str,
        format: str = 'hdf5'
    ) -> str:
        """Export dataset for reproducibility.
        
        Args:
            dataset: Dataset dictionary
            output_path: Output file path
            format: Export format
            
        Returns:
            Path to saved file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'hdf5':
            with h5py.File(output_path, 'w') as f:
                f.attrs['export_date'] = datetime.now().isoformat()
                
                for key, value in dataset.items():
                    if isinstance(value, torch.Tensor):
                        f.create_dataset(key, data=value.numpy())
                    elif isinstance(value, dict):
                        # Handle nested dictionaries (e.g., vocabulary)
                        grp = f.create_group(key)
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, (int, float, str)):
                                grp.attrs[sub_key] = sub_value
                    else:
                        f.attrs[key] = value
        
        elif format == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump(dataset, f)
        
        return str(output_path)


class ReportGenerator:
    """Generate comprehensive analysis reports."""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.figure_exporter = FigureExporter(config)
        self.data_exporter = DataExporter()
    
    def generate_html_report(
        self,
        analysis_results: Dict[str, Any],
        figures: Dict[str, Union[plt.Figure, go.Figure]],
        output_path: str,
        title: str = "Positional Encoding Analysis Report"
    ) -> str:
        """Generate comprehensive HTML report.
        
        Args:
            analysis_results: Analysis results dictionary
            figures: Dictionary of figures
            output_path: Output HTML file path
            title: Report title
            
        Returns:
            Path to generated HTML report
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert figures to base64 for embedding
        embedded_figures = {}
        for name, fig in figures.items():
            if isinstance(fig, plt.Figure):
                img_buffer = io.BytesIO()
                fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
                img_buffer.seek(0)
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
                embedded_figures[name] = f"data:image/png;base64,{img_base64}"
            elif isinstance(fig, go.Figure):
                embedded_figures[name] = fig.to_html(include_plotlyjs='inline', div_id=f"div_{name}")
        
        # Generate HTML content
        html_content = self._generate_html_content(title, analysis_results, embedded_figures)
        
        # Save HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return str(output_path)
    
    def _generate_html_content(
        self,
        title: str,
        analysis_results: Dict[str, Any],
        embedded_figures: Dict[str, str]
    ) -> str:
        """Generate HTML content for the report."""
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1 {{
                    color: #2c3e50;
                    text-align: center;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    border-left: 4px solid #3498db;
                    padding-left: 10px;
                    margin-top: 30px;
                }}
                .metric-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 15px;
                    margin: 20px 0;
                }}
                .metric-card {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    border-left: 4px solid #3498db;
                }}
                .metric-name {{
                    font-weight: bold;
                    color: #2c3e50;
                }}
                .metric-value {{
                    font-size: 1.2em;
                    color: #27ae60;
                    margin-top: 5px;
                }}
                .figure-container {{
                    margin: 20px 0;
                    text-align: center;
                }}
                .figure-title {{
                    font-weight: bold;
                    margin-bottom: 10px;
                    color: #2c3e50;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                .timestamp {{
                    text-align: center;
                    color: #7f8c8d;
                    font-size: 0.9em;
                    margin-top: 30px;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                }}
                tr:nth-child(even) {{
                    background-color: #f2f2f2;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>{title}</h1>
                
                {self._generate_summary_section(analysis_results)}
                
                {self._generate_metrics_section(analysis_results)}
                
                {self._generate_figures_section(embedded_figures)}
                
                <div class="timestamp">
                    Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_summary_section(self, analysis_results: Dict[str, Any]) -> str:
        """Generate summary section of the report."""
        summary_html = """
        <h2>📊 Summary</h2>
        <p>This report presents a comprehensive analysis of positional encoding methods and their effects on transformer attention patterns.</p>
        """
        
        if 'model_config' in analysis_results:
            config = analysis_results['model_config']
            summary_html += f"""
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-name">Model Dimension</div>
                    <div class="metric-value">{config.get('d_model', 'N/A')}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">Attention Heads</div>
                    <div class="metric-value">{config.get('n_heads', 'N/A')}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">Sequence Length</div>
                    <div class="metric-value">{config.get('max_seq_len', 'N/A')}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-name">Encoding Type</div>
                    <div class="metric-value">{config.get('encoding_type', 'N/A')}</div>
                </div>
            </div>
            """
        
        return summary_html
    
    def _generate_metrics_section(self, analysis_results: Dict[str, Any]) -> str:
        """Generate metrics section of the report."""
        metrics_html = "<h2>📈 Analysis Metrics</h2>"
        
        # Process different types of metrics
        for section_name, section_data in analysis_results.items():
            if section_name in ['figures', 'model_config']:
                continue
            
            metrics_html += f"<h3>{section_name.replace('_', ' ').title()}</h3>"
            
            if isinstance(section_data, dict):
                metrics_html += '<div class="metric-grid">'
                for metric_name, metric_value in section_data.items():
                    if isinstance(metric_value, (int, float)):
                        formatted_value = f"{metric_value:.4f}" if isinstance(metric_value, float) else str(metric_value)
                        metrics_html += f"""
                        <div class="metric-card">
                            <div class="metric-name">{metric_name.replace('_', ' ').title()}</div>
                            <div class="metric-value">{formatted_value}</div>
                        </div>
                        """
                metrics_html += '</div>'
        
        return metrics_html
    
    def _generate_figures_section(self, embedded_figures: Dict[str, str]) -> str:
        """Generate figures section of the report."""
        figures_html = "<h2>📊 Visualizations</h2>"
        
        for figure_name, figure_data in embedded_figures.items():
            figures_html += f"""
            <div class="figure-container">
                <div class="figure-title">{figure_name.replace('_', ' ').title()}</div>
                """
            
            if figure_data.startswith('data:image'):
                # Embedded image
                figures_html += f'<img src="{figure_data}" alt="{figure_name}">'
            else:
                # Plotly HTML
                figures_html += figure_data
            
            figures_html += "</div>"
        
        return figures_html
    
    def generate_complete_analysis_package(
        self,
        analysis_results: Dict[str, Any],
        figures: Dict[str, Union[plt.Figure, go.Figure]],
        output_dir: str,
        package_name: str = "positional_encoding_analysis"
    ) -> str:
        """Generate complete analysis package with all outputs.
        
        Args:
            analysis_results: Analysis results
            figures: Dictionary of figures
            output_dir: Output directory
            package_name: Package name
            
        Returns:
            Path to analysis package directory
        """
        package_dir = Path(output_dir) / package_name
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Export figures
        figures_dir = package_dir / "figures"
        self.figure_exporter.export_multiple_figures(figures, figures_dir, "png")
        self.figure_exporter.export_multiple_figures(figures, figures_dir, "pdf")
        
        # Create PDF report with matplotlib figures
        matplotlib_figures = {k: v for k, v in figures.items() if isinstance(v, plt.Figure)}
        if matplotlib_figures:
            self.figure_exporter.create_pdf_report(
                matplotlib_figures,
                package_dir / "figures_report.pdf",
                "Positional Encoding Analysis"
            )
        
        # Export data
        data_dir = package_dir / "data"
        self.data_exporter.export_metrics(analysis_results, data_dir / "metrics.json")
        
        # Generate HTML report
        self.generate_html_report(
            analysis_results,
            figures,
            package_dir / "analysis_report.html"
        )
        
        # Create README
        readme_content = f"""# {package_name.replace('_', ' ').title()}

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Contents

- `analysis_report.html`: Interactive HTML report with all visualizations and metrics
- `figures_report.pdf`: PDF report with all matplotlib figures
- `figures/`: Individual figure files in PNG and PDF format
- `data/`: Exported analysis data and metrics

## Usage

Open `analysis_report.html` in a web browser to view the complete interactive analysis.

"""
        
        with open(package_dir / "README.md", 'w') as f:
            f.write(readme_content)
        
        return str(package_dir)


# Convenience functions
def export_visualization(
    figure: Union[plt.Figure, go.Figure],
    filepath: str,
    format: Optional[str] = None,
    config: Optional[VisualizationConfig] = None
) -> str:
    """Export single visualization.
    
    Args:
        figure: Figure to export
        filepath: Output file path
        format: Export format
        config: Visualization config
        
    Returns:
        Path to saved file
    """
    if config is None:
        config = VisualizationConfig()
    
    exporter = FigureExporter(config)
    
    if isinstance(figure, plt.Figure):
        return exporter.export_matplotlib_figure(figure, filepath, format)
    elif isinstance(figure, go.Figure):
        return exporter.export_plotly_figure(figure, filepath, format)
    else:
        raise ValueError("Unknown figure type")


def save_attention_weights(
    attention_weights: List[torch.Tensor],
    filepath: str,
    tokens: Optional[List[str]] = None,
    format: str = 'hdf5'
) -> str:
    """Save attention weights to file.
    
    Args:
        attention_weights: List of attention weight tensors
        filepath: Output file path
        tokens: Token strings
        format: Export format
        
    Returns:
        Path to saved file
    """
    exporter = DataExporter()
    return exporter.export_attention_weights(attention_weights, filepath, tokens, format=format)


def generate_analysis_report(
    analysis_results: Dict[str, Any],
    figures: Dict[str, Union[plt.Figure, go.Figure]],
    output_path: str,
    title: str = "Analysis Report",
    config: Optional[VisualizationConfig] = None
) -> str:
    """Generate analysis report.
    
    Args:
        analysis_results: Analysis results
        figures: Dictionary of figures
        output_path: Output file path
        title: Report title
        config: Visualization config
        
    Returns:
        Path to generated report
    """
    if config is None:
        config = VisualizationConfig()
    
    generator = ReportGenerator(config)
    return generator.generate_html_report(analysis_results, figures, output_path, title)
