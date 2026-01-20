"""
Export utilities for DOE Toolkit.

Provides functions for exporting projects and generating comprehensive
HTML reports with all analysis results, plots, and tables.
"""

from typing import Dict, Optional, List
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime
import base64
from io import BytesIO


def generate_html_report() -> str:
    """
    Generate comprehensive HTML report of entire project.
    
    Includes:
    - Project summary
    - Factor definitions
    - Design matrix
    - Response data
    - ANOVA tables for all responses
    - Effect plots
    - Diagnostic plots
    - Augmentation summary (if applicable)
    - Optimization results (if applicable)
    
    Returns
    -------
    str
        Complete HTML document
    """
    html_parts = []
    
    # HTML header
    html_parts.append(_get_html_header())
    
    # Title and metadata
    html_parts.append(_get_project_metadata())
    
    # Table of contents
    html_parts.append(_get_table_of_contents())
    
    # Section 1: Project Overview
    html_parts.append(_get_project_overview())
    
    # Section 2: Factor Definitions
    if st.session_state.get('factors'):
        html_parts.append(_get_factors_section())
    
    # Section 3: Design Matrix
    if st.session_state.get('design') is not None:
        html_parts.append(_get_design_section())
    
    # Section 4: Response Data
    if st.session_state.get('responses'):
        html_parts.append(_get_responses_section())
    
    # Section 5: Analysis Results
    if st.session_state.get('fitted_models'):
        html_parts.append(_get_analysis_section())
    
    # Section 6: Quality Assessment (if diagnostics exist)
    if st.session_state.get('quality_report'):
        html_parts.append(_get_quality_section())
    
    # Section 7: Augmentation Summary (if applicable)
    if st.session_state.get('augmented_design'):
        html_parts.append(_get_augmentation_section())
    
    # Section 8: Optimization Results (if applicable)
    if st.session_state.get('optimization_results'):
        html_parts.append(_get_optimization_section())
    
    # Footer
    html_parts.append(_get_html_footer())
    
    return '\n'.join(html_parts)


def _get_html_header() -> str:
    """Generate HTML document header with CSS."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DOE Toolkit Analysis Report</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-radius: 8px;
        }
        
        h1 {
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        
        h2 {
            color: #34495e;
            margin-top: 40px;
            margin-bottom: 20px;
            padding-bottom: 8px;
            border-bottom: 2px solid #ecf0f1;
        }
        
        h3 {
            color: #555;
            margin-top: 25px;
            margin-bottom: 15px;
        }
        
        h4 {
            color: #666;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        
        .metadata {
            background: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 30px;
            font-size: 0.9em;
        }
        
        .metadata-row {
            margin-bottom: 5px;
        }
        
        .metadata-label {
            font-weight: bold;
            display: inline-block;
            width: 150px;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 0.9em;
        }
        
        th {
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
            font-weight: 600;
        }
        
        td {
            padding: 10px 12px;
            border-bottom: 1px solid #ecf0f1;
        }
        
        tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        
        tr:hover {
            background-color: #e8f4f8;
        }
        
        .numeric {
            text-align: right;
            font-family: 'Courier New', monospace;
        }
        
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .metric-card {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }
        
        .metric-label {
            font-size: 0.85em;
            color: #777;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 1.5em;
            font-weight: bold;
            color: #2c3e50;
        }
        
        .warning {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }
        
        .success {
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }
        
        .info {
            background: #d1ecf1;
            border-left: 4px solid #17a2b8;
            padding: 15px;
            margin: 15px 0;
            border-radius: 4px;
        }
        
        .toc {
            background: #f8f9fa;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        
        .toc ul {
            list-style-position: inside;
            padding-left: 20px;
        }
        
        .toc li {
            margin: 8px 0;
        }
        
        .toc a {
            color: #3498db;
            text-decoration: none;
        }
        
        .toc a:hover {
            text-decoration: underline;
        }
        
        .page-break {
            page-break-after: always;
        }
        
        .formula {
            background: #f8f9fa;
            padding: 10px;
            margin: 10px 0;
            border-left: 3px solid #3498db;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
        }
        
        .significant {
            background-color: #ffffcc;
            font-weight: bold;
        }
        
        footer {
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            text-align: center;
            color: #777;
            font-size: 0.85em;
        }
        
        @media print {
            body {
                background: white;
                padding: 0;
            }
            
            .container {
                box-shadow: none;
                padding: 20px;
            }
            
            .page-break {
                page-break-after: always;
            }
        }
    </style>
</head>
<body>
    <div class="container">
"""


def _get_project_metadata() -> str:
    """Generate project metadata section."""
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    
    design_type = st.session_state.get('design_type', 'Unknown')
    n_factors = len(st.session_state.get('factors', []))
    n_responses = len(st.session_state.get('responses', {}))
    
    design = st.session_state.get('design')
    n_runs = len(design) if design is not None else 0
    
    html = f"""
        <h1>🔬 DOE Toolkit Analysis Report</h1>
        
        <div class="metadata">
            <div class="metadata-row">
                <span class="metadata-label">Generated:</span>
                <span>{timestamp}</span>
            </div>
            <div class="metadata-row">
                <span class="metadata-label">Design Type:</span>
                <span>{design_type.replace('_', ' ').title()}</span>
            </div>
            <div class="metadata-row">
                <span class="metadata-label">Number of Factors:</span>
                <span>{n_factors}</span>
            </div>
            <div class="metadata-row">
                <span class="metadata-label">Number of Runs:</span>
                <span>{n_runs}</span>
            </div>
            <div class="metadata-row">
                <span class="metadata-label">Number of Responses:</span>
                <span>{n_responses}</span>
            </div>
        </div>
    """
    
    return html


def _get_table_of_contents() -> str:
    """Generate table of contents."""
    toc_items = []
    
    toc_items.append('<li><a href="#overview">Project Overview</a></li>')
    
    if st.session_state.get('factors'):
        toc_items.append('<li><a href="#factors">Factor Definitions</a></li>')
    
    if st.session_state.get('design') is not None:
        toc_items.append('<li><a href="#design">Design Matrix</a></li>')
    
    if st.session_state.get('responses'):
        toc_items.append('<li><a href="#responses">Response Data</a></li>')
    
    if st.session_state.get('fitted_models'):
        toc_items.append('<li><a href="#analysis">Analysis Results</a></li>')
    
    if st.session_state.get('quality_report'):
        toc_items.append('<li><a href="#quality">Quality Assessment</a></li>')
    
    if st.session_state.get('augmented_design'):
        toc_items.append('<li><a href="#augmentation">Augmentation Summary</a></li>')
    
    if st.session_state.get('optimization_results'):
        toc_items.append('<li><a href="#optimization">Optimization Results</a></li>')
    
    html = f"""
        <div class="toc">
            <h3>Table of Contents</h3>
            <ul>
                {''.join(toc_items)}
            </ul>
        </div>
    """
    
    return html


def _get_project_overview() -> str:
    """Generate project overview section."""
    html = '<h2 id="overview">1. Project Overview</h2>'
    
    design_metadata = st.session_state.get('design_metadata', {})
    
    if design_metadata:
        html += '<div class="metric-grid">'
        
        for key, value in design_metadata.items():
            if key not in ['generators', 'alias_structure']:  # Skip complex nested data
                html += f"""
                <div class="metric-card">
                    <div class="metric-label">{key.replace('_', ' ').title()}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """
        
        html += '</div>'
    
    return html


def _get_factors_section() -> str:
    """Generate factors section."""
    factors = st.session_state['factors']
    
    html = '<h2 id="factors">2. Factor Definitions</h2>'
    
    # Create factors table
    factors_data = []
    for factor in factors:
        factors_data.append({
            'Name': factor.name,
            'Type': factor.factor_type.value.replace('_', ' ').title(),
            'Changeability': factor.changeability.value.replace('_', ' ').title(),
            'Levels/Range': _format_factor_levels(factor),
            'Units': factor.units if factor.units else '—'
        })
    
    df = pd.DataFrame(factors_data)
    html += _dataframe_to_html(df)
    
    return html


def _format_factor_levels(factor) -> str:
    """Format factor levels for display."""
    if factor.is_continuous():
        return f"[{factor.min_value}, {factor.max_value}]"
    else:
        return ', '.join(str(level) for level in factor.levels)


def _get_design_section() -> str:
    """Generate design matrix section."""
    design = st.session_state['design']
    
    html = '<h2 id="design">3. Design Matrix</h2>'
    
    # Show design summary
    html += f'<p><strong>Total Runs:</strong> {len(design)}</p>'
    
    # Show first 50 runs (for large designs)
    if len(design) <= 50:
        html += _dataframe_to_html(design)
    else:
        html += '<div class="info">Design has more than 50 runs. Showing first 50 rows:</div>'
        html += _dataframe_to_html(design.head(50))
        html += f'<p><em>... and {len(design) - 50} more rows</em></p>'
    
    return html


def _get_responses_section() -> str:
    """Generate response data section."""
    responses = st.session_state['responses']
    design = st.session_state['design']
    
    html = '<h2 id="responses">4. Response Data</h2>'
    
    # Create combined DataFrame
    combined_df = design.copy()
    for response_name, response_data in responses.items():
        combined_df[response_name] = response_data
    
    # Show summary statistics
    html += '<h3>Summary Statistics</h3>'
    
    stats_data = []
    for response_name in responses.keys():
        data = combined_df[response_name]
        stats_data.append({
            'Response': response_name,
            'Mean': f"{data.mean():.3f}",
            'Std Dev': f"{data.std():.3f}",
            'Min': f"{data.min():.3f}",
            'Max': f"{data.max():.3f}",
            'N': len(data)
        })
    
    stats_df = pd.DataFrame(stats_data)
    html += _dataframe_to_html(stats_df)
    
    # Show data table (first 50 rows)
    html += '<h3>Experimental Data</h3>'
    
    if len(combined_df) <= 50:
        html += _dataframe_to_html(combined_df)
    else:
        html += '<div class="info">Showing first 50 rows:</div>'
        html += _dataframe_to_html(combined_df.head(50))
        html += f'<p><em>... and {len(combined_df) - 50} more rows</em></p>'
    
    return html


def _get_analysis_section() -> str:
    """Generate analysis results section."""
    fitted_models = st.session_state['fitted_models']
    
    html = '<h2 id="analysis">5. Analysis Results</h2>'
    
    for response_name, results in fitted_models.items():
        html += f'<div class="page-break"></div>'
        html += f'<h3>Response: {response_name}</h3>'
        
        # Model summary metrics
        html += '<div class="metric-grid">'
        html += f"""
        <div class="metric-card">
            <div class="metric-label">R²</div>
            <div class="metric-value">{results.r_squared:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Adjusted R²</div>
            <div class="metric-value">{results.adj_r_squared:.4f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">RMSE</div>
            <div class="metric-value">{results.rmse:.4f}</div>
        </div>
        """
        html += '</div>'
        
        # Model formula
        html += f'<h4>Model Formula</h4>'
        html += f'<div class="formula">{results.model_formula}</div>'
        
        # ANOVA table
        html += '<h4>ANOVA Table</h4>'
        html += _dataframe_to_html(results.anova_table, highlight_significant=True)
        
        # Effect estimates
        html += '<h4>Effect Estimates</h4>'
        html += _dataframe_to_html(results.effects_table, highlight_significant=True)
        
        # Diagnostics
        if hasattr(results, 'lack_of_fit_p'):
            if results.lack_of_fit_p < 0.05:
                html += f'<div class="warning"><strong>⚠️ Lack of Fit:</strong> p-value = {results.lack_of_fit_p:.4f} (significant)</div>'
            else:
                html += f'<div class="success"><strong>✓ Lack of Fit:</strong> p-value = {results.lack_of_fit_p:.4f} (not significant)</div>'
    
    return html


def _get_quality_section() -> str:
    """Generate quality assessment section."""
    quality_report = st.session_state['quality_report']
    
    html = '<h2 id="quality">6. Design Quality Assessment</h2>'
    
    # Overall summary
    html += '<h3>Overall Assessment</h3>'
    
    if quality_report.critical_issues:
        html += '<div class="warning">'
        html += '<h4>Critical Issues</h4><ul>'
        for issue in quality_report.critical_issues:
            html += f'<li>{issue}</li>'
        html += '</ul></div>'
    
    if quality_report.warnings:
        html += '<div class="info">'
        html += '<h4>Warnings</h4><ul>'
        for warning in quality_report.warnings:
            html += f'<li>{warning}</li>'
        html += '</ul></div>'
    
    if quality_report.satisfactory_aspects:
        html += '<div class="success">'
        html += '<h4>Satisfactory Aspects</h4><ul>'
        for aspect in quality_report.satisfactory_aspects:
            html += f'<li>{aspect}</li>'
        html += '</ul></div>'
    
    # Per-response quality
    html += '<h3>Per-Response Quality</h3>'
    
    for response_name, assessment in quality_report.response_quality.items():
        html += f'<h4>{response_name}: {assessment.overall_grade}</h4>'
        
        if assessment.issues:
            html += '<ul>'
            for issue in assessment.issues:
                html += f'<li><strong>{issue.category}:</strong> {issue.description}</li>'
            html += '</ul>'
    
    return html


def _get_augmentation_section() -> str:
    """Generate augmentation summary section."""
    augmented = st.session_state['augmented_design']
    selected_plan = st.session_state.get('selected_plan')
    
    html = '<h2 id="augmentation">7. Design Augmentation</h2>'
    
    if selected_plan:
        html += f'<h3>Selected Strategy: {selected_plan.plan_name}</h3>'
        html += f'<p><strong>Method:</strong> {selected_plan.strategy}</p>'
        html += f'<p><strong>Runs Added:</strong> {selected_plan.n_runs_to_add}</p>'
        
        html += '<h4>Expected Improvements</h4><ul>'
        for metric, improvement in selected_plan.expected_improvements.items():
            html += f'<li><strong>{metric}:</strong> {improvement}</li>'
        html += '</ul>'
    
    html += '<h3>Augmentation Results</h3>'
    
    html += '<div class="metric-grid">'
    html += f"""
    <div class="metric-card">
        <div class="metric-label">Original Runs</div>
        <div class="metric-value">{augmented.n_runs_original}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Runs Added</div>
        <div class="metric-value">{augmented.n_runs_added}</div>
    </div>
    <div class="metric-card">
        <div class="metric-label">Total Runs</div>
        <div class="metric-value">{augmented.n_runs_total}</div>
    </div>
    """
    
    if augmented.d_efficiency:
        html += f"""
        <div class="metric-card">
            <div class="metric-label">D-Efficiency</div>
            <div class="metric-value">{augmented.d_efficiency:.1f}%</div>
        </div>
        """
    
    html += '</div>'
    
    # Show achieved improvements
    if augmented.achieved_improvements:
        html += '<h4>Achieved Improvements</h4><ul>'
        for metric, improvement in augmented.achieved_improvements.items():
            html += f'<li><strong>{metric}:</strong> {improvement}</li>'
        html += '</ul>'
    
    # Show new runs
    html += '<h3>New Runs to Execute</h3>'
    html += _dataframe_to_html(augmented.new_runs_only)
    
    return html


def _get_optimization_section() -> str:
    """Generate optimization results section."""
    opt_result = st.session_state['optimization_results']
    
    html = '<h2 id="optimization">8. Optimization Results</h2>'
    
    if opt_result.success:
        html += '<div class="success">✓ Optimization converged successfully</div>'
    else:
        html += f'<div class="warning">⚠️ Optimization warning: {opt_result.message}</div>'
    
    html += '<h3>Optimal Settings</h3>'
    
    settings_data = []
    for factor_name, value in opt_result.optimal_settings.items():
        settings_data.append({
            'Factor': factor_name,
            'Optimal Value': f"{value:.4f}"
        })
    
    settings_df = pd.DataFrame(settings_data)
    html += _dataframe_to_html(settings_df)
    
    html += '<h3>Predicted Response</h3>'
    
    html += '<div class="metric-grid">'
    html += f"""
    <div class="metric-card">
        <div class="metric-label">Predicted Value</div>
        <div class="metric-value">{opt_result.predicted_response:.4f}</div>
    </div>
    """
    
    ci_lower, ci_upper = opt_result.confidence_interval
    html += f"""
    <div class="metric-card">
        <div class="metric-label">95% Confidence Interval</div>
        <div class="metric-value">[{ci_lower:.4f}, {ci_upper:.4f}]</div>
    </div>
    """
    
    pi_lower, pi_upper = opt_result.prediction_interval
    html += f"""
    <div class="metric-card">
        <div class="metric-label">95% Prediction Interval</div>
        <div class="metric-value">[{pi_lower:.4f}, {pi_upper:.4f}]</div>
    </div>
    """
    
    html += '</div>'
    
    html += f'<p><strong>Objective Value:</strong> {opt_result.objective_value:.6f}</p>'
    html += f'<p><strong>Iterations:</strong> {opt_result.n_iterations}</p>'
    
    return html


def _get_html_footer() -> str:
    """Generate HTML footer."""
    return """
        <footer>
            <p>Generated by DOE Toolkit v0.1.0</p>
            <p>Free, open-source Design of Experiments software</p>
            <p><a href="https://github.com/bpimentel3/doe-toolkit">github.com/bpimentel3/doe-toolkit</a></p>
        </footer>
    </div>
</body>
</html>
"""


def _dataframe_to_html(df: pd.DataFrame, highlight_significant: bool = False) -> str:
    """
    Convert DataFrame to HTML table.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to convert
    highlight_significant : bool
        If True, highlight rows where p-value < 0.05
    
    Returns
    -------
    str
        HTML table string
    """
    html = '<table>'
    
    # Header
    html += '<tr>'
    for col in df.columns:
        html += f'<th>{col}</th>'
    html += '</tr>'
    
    # Rows
    for _, row in df.iterrows():
        # Check if significant (if p-value column exists)
        is_significant = False
        if highlight_significant and 'p-value' in df.columns:
            try:
                p_val = float(row['p-value'])
                is_significant = p_val < 0.05
            except (ValueError, TypeError):
                pass
        
        row_class = ' class="significant"' if is_significant else ''
        html += f'<tr{row_class}>'
        
        for col in df.columns:
            value = row[col]
            
            # Format numeric values
            if isinstance(value, (int, float, np.number)):
                if abs(value) < 0.001 and value != 0:
                    formatted = f'{value:.2e}'
                else:
                    formatted = f'{value:.4f}' if isinstance(value, float) else str(value)
                html += f'<td class="numeric">{formatted}</td>'
            else:
                html += f'<td>{value}</td>'
        
        html += '</tr>'
    
    html += '</table>'
    
    return html
