import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

st.set_page_config(
    page_title="Numerical Differentiation Visualization",
    page_icon="📈",
    layout="wide"
)

# Function definitions
functions = {
    'exp': {
        'fn': lambda x: np.exp(x),
        'derivative': lambda x: np.exp(x),
        'second_derivative': lambda x: np.exp(x),
        'third_derivative': lambda x: np.exp(x),
        'label': 'e^x'
    },
    'sin': {
        'fn': lambda x: np.sin(x),
        'derivative': lambda x: np.cos(x),
        'second_derivative': lambda x: -np.sin(x),
        'third_derivative': lambda x: -np.cos(x),
        'label': 'sin(x)'
    },
    'cubic': {
        'fn': lambda x: x**3,
        'derivative': lambda x: 3 * x**2,
        'second_derivative': lambda x: 6 * x,
        'third_derivative': lambda x: 6,
        'label': 'x^3'
    }
}

def forward_difference(f, x, h):
    return (f(x + h) - f(x)) / h

def central_difference(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

def backward_difference(f, x, h):
    return (f(x) - f(x - h)) / h

def get_step_by_step_calculation(method, f, x, h, exact_derivative, second_derivative,third_derivative):
    steps = []
    if method == "Forward":
        fx = f(x)
        fxh = f(x + h)
        numerical_derivative = (fxh - fx) / h
        truncation_error = abs(-(h / 2) * second_derivative(x + h))
        error = abs(numerical_derivative - exact_derivative)

        steps = [
            f"1. Calculate f(x) at x = {x:.3f}:",
            f"   f({x:.3f}) = {fx:.3f}",
            f"2. Calculate f(x + h) at x + h = {x + h:.3f}:",
            f"   f({x + h:.3f}) = {fxh:.3f}",
            f"3. Apply forward difference formula:",
            f"   f'(x) ≈ [f(x + h) - f(x)] / h",
            f"   f'({x:.3f}) ≈ [{fxh:.3f} - {fx:.3f}] / {h:.3f}",
            f"   f'({x:.3f}) ≈ {numerical_derivative:.3f}",
            f"4. Calculate truncation error using second derivative:",
            f"   Error ≈ (h / 2) * f''(x + h)",
            f"   f''({x + h:.3f}) = {second_derivative(x + h):.3f}",
            f"   Truncation Error ≈ ({h:.3f} / 2) * {second_derivative(x + h):.3f}",
            f"   |Truncation Error| ≈ {truncation_error:.3f}",
            f"5. Calculate absolute error:",
            f"   Exact Derivative at x = {x:.3f}: f'({x:.3f}) = {exact_derivative:.3f}",
            f"   |Exact - Approximate| = |{exact_derivative:.3f} - {numerical_derivative:.3f}|",
            f"   Absolute Error = {error:.3f}"
        ]
    elif method == "Central":
        fxph = f(x + h)
        fxmh = f(x - h)
        numerical_derivative = (fxph - fxmh) / (2 * h)
        truncation_error = abs(-(h**2 / 6) * third_derivative(x))
        error = abs(numerical_derivative - exact_derivative)

        steps = [
            f"1. Calculate f(x + h) at x + h = {x + h:.3f}:",
            f"   f({x + h:.3f}) = {fxph:.3f}",
            f"2. Calculate f(x - h) at x - h = {x - h:.3f}:",
            f"   f({x - h:.3f}) = {fxmh:.3f}",
            f"3. Apply central difference formula:",
            f"   f'(x) ≈ [f(x + h) - f(x - h)] / (2h)",
            f"   f'({x:.3f}) ≈ [{fxph:.3f} - {fxmh:.3f}] / (2 * {h:.3f})",
            f"   f'({x:.3f}) ≈ {numerical_derivative:.3f}",
            f"4. Calculate truncation error using third derivative:",
            f"   Error ≈ -(h^2 / 6) * f'''(x)",
            f"   f''({x:.3f}) = {third_derivative(x):.3f}",
            f"   Truncation Error ≈ -({h:.3f}^2 / 6) * {third_derivative(x):.3f}",
            f"   |Truncation Error| ≈ {truncation_error:.3f}",
            f"5. Calculate absolute error:",
            f"   Exact Derivative at x = {x:.3f}: f'({x:.3f}) = {exact_derivative:.3f}",
            f"   |Exact - Approximate| = |{exact_derivative:.3f} - {numerical_derivative:.3f}|",
            f"   Absolute Error = {error:.3f}"
            
        ]
    elif method == "Backward":
        fx = f(x)
        fxmh = f(x - h)
        numerical_derivative = (fx - fxmh) / h
        truncation_error = abs((h / 2) * second_derivative(x - h))
        error = abs(numerical_derivative - exact_derivative)
        
        steps = [
            f"1. Calculate f(x) at x = {x:.3f}:",
            f"   f({x:.3f}) = {fx:.3f}",
            f"2. Calculate f(x - h) at x - h = {x - h:.3f}:",
            f"   f({x - h:.3f}) = {fxmh:.3f}",
            f"3. Apply backward difference formula:",
            f"   f'(x) ≈ [f(x) - f(x - h)] / h",
            f"   f'({x:.3f}) ≈ [{fx:.3f} - {fxmh:.3f}] / {h:.3f}",
            f"   f'({x:.3f}) ≈ {numerical_derivative:.3f}",
            f"4. Calculate truncation error using second derivative:",
            f"   Error ≈ (h / 2) * f''(x - h)",
            f"   f''({x - h:.3f}) = {second_derivative(x - h):.3f}",
            f"   Truncation Error ≈ ({h:.3f} / 2) * {second_derivative(x - h):.3f}",
            f"   |Truncation Error| ≈ {truncation_error:.3f}",
            f"5. Calculate absolute error:",
            f"   Exact Derivative at x = {x:.3f}: f'({x:.3f}) = {exact_derivative:.3f}",
            f"   |Exact - Approximate| = |{exact_derivative:.3f} - {numerical_derivative:.3f}|",
            f"   Absolute Error = {error:.3f}"
            
        ]
    return steps

# Streamlit app
st.title("Numerical Differentiation Visualization")

# Sidebar controls
with st.sidebar:
    st.header("Controls")

    # Function selection
    selected_function = st.radio(
        "Select Function",
        options=['exp', 'sin', 'cubic'],
        format_func=lambda x: functions[x]['label']
    )
    x_point = st.number_input("Test Point (x)")
    step_size = st.number_input("Test Step Size (h)")

    # Method selection
    st.header("Methods")
    show_forward = st.checkbox("Forward Difference", value=True)
    show_central = st.checkbox("Central Difference", value=True)
    show_backward = st.checkbox("Backward Difference", value=True)

# Get function and its derivative
f = functions[selected_function]['fn']
f_prime = functions[selected_function]['derivative']
f_double_prime = functions[selected_function]['second_derivative']
f_triple_prime = functions[selected_function]['third_derivative']

# Calculate derivatives and errors
exact_derivative = f_prime(x_point)
results = {}

if show_forward:
    forward_result = forward_difference(f, x_point, step_size)
    forward_error =  abs(forward_result - exact_derivative)
    results['Forward'] = {'value': forward_result, 'error': forward_error}

if show_central:
    central_result = central_difference(f, x_point, step_size)
    central_error = abs(central_result - exact_derivative)
    results['Central'] = {'value': central_result, 'error': central_error}

if show_backward:
    backward_result = backward_difference(f, x_point, step_size)
    backward_error = abs(backward_result - exact_derivative)
    results['Backward'] = {'value': backward_result, 'error': backward_error}

# Create visualization
x = np.linspace(-2, 2, 200)
y = f(x)

# Create subplot with function and derivative
fig = make_subplots(
    rows=2, 
    cols=1, 
    subplot_titles=('Function and Approximations', 'Error Analysis'),
    row_heights=[0.7, 0.7]
)


# Plot original function
fig.add_trace(
    go.Scatter(x=x, y=y, name='Function', line=dict(color='black')),
    row=1, col=1
)

# Plot point of interest
fig.add_trace(
    go.Scatter(x=[x_point], y=[f(x_point)], 
               mode='markers', name='Point',
               marker=dict(size=10, color='black')),
    row=1, col=1
)

# Plot exact derivative tangent line
tangent_x = np.array([x_point - 0.5, x_point + 0.5])
tangent_y = f(x_point) + exact_derivative * (tangent_x - x_point)
fig.add_trace(
    go.Scatter(x=tangent_x, y=tangent_y, 
               name='Exact Derivative',
               line=dict(color='orange', dash='dash')),
    row=1, col=1
)

# Colors for different methods
colors = {'Forward': 'red', 'Central': 'blue', 'Backward': 'purple'}

# Plot numerical approximations
for method, data in results.items():
    if method == 'Forward':
        h_points = [x_point, x_point + step_size]
    elif method == 'Backward':
        h_points = [x_point - step_size, x_point]
    else:  # Central
        h_points = [x_point - step_size, x_point + step_size]

    # Plot vertical lines at h points
    for h_point in h_points:
        fig.add_trace(
            go.Scatter(x=[h_point, h_point], 
                      y=[0, f(h_point)],
                      name=f'{method} h-line',
                      line=dict(color=colors[method], dash='dot'),
                      showlegend=False),
            row=1, col=1
        )

    # Plot approximation points
    fig.add_trace(
        go.Scatter(x=h_points, y=[f(h) for h in h_points],
                  mode='markers',
                  name=f'{method} Points',
                  marker=dict(color=colors[method], size=8)),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=h_points, y=[f(h) for h in h_points],
                  mode='markers',
                  name=f'{method} Points',
                  marker=dict(color=colors[method], size=8)),
        row=1, col=1
    )
    
    # Add approximate derivative lines
    approx_tangent_x = np.array([x_point - 0.5, x_point + 0.5])
    approx_tangent_y = f(x_point) + data['value'] * (approx_tangent_x - x_point)
    fig.add_trace(
        go.Scatter(x=approx_tangent_x, y=approx_tangent_y,
                  name=f'{method} Approximation',
                  line=dict(color=colors[method])),
        row=1, col=1
    )

fig.update_layout(
    height=800, 
    showlegend=True,
)

# Add gridlines specifically to the function plot (row 1)
fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128, 128, 128, 0.2)',
    zeroline=True,
    zerolinewidth=1.5,
    zerolinecolor='rgba(128, 128, 128, 0.5)',
    row=1, 
    col=1
)

fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(128, 128, 128, 0.2)',
    zeroline=True,
    zerolinewidth=1.5,
    zerolinecolor='rgba(128, 128, 128, 0.5)',
    row=1, 
    col=1
)

# Plot error bars
error_data = pd.DataFrame([
    {'Method': method, 'Error': data['error']}
    for method, data in results.items()
])

fig.add_trace(
    go.Bar(x=error_data['Method'], 
           y=error_data['Error'],
           name='Error',
           marker_color=[colors[method] for method in error_data['Method']]),
    row=2, col=1
)

# Update layout
fig.update_layout(height=800, showlegend=True)
fig.update_xaxes(title_text="x", row=2, col=1)
fig.update_yaxes(title_text="Error", row=2, col=1)

# Display plot
st.plotly_chart(fig, use_container_width=True)

# Display results in a clean grid
st.header("Numerical Results")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Exact Value", f"{exact_derivative:.3f}")

for i, (method, data) in enumerate(results.items(), start=1):
    with [col2, col3, col4][i-1]:
        st.metric(
            method,
            f"{data['value']:.3f}",
            f"Error: {data['error']:.3f}"
        )
st.header("Error Analysis Table")

if 'table_data' not in st.session_state:
    st.session_state.table_data = []
    st.session_state.previous_h = None

# Check if step size has changed
if st.session_state.previous_h != step_size:
    # Add new row for current h value
    row_data = {'Step Size (h)': f"{step_size:.3f}"}
    
    if show_forward:
        fwd_deriv = forward_difference(f, x_point, step_size)
        fwd_trunc = abs(-(step_size / 2) * f_double_prime(x_point + step_size))
        row_data['Forward f\'(x)'] = f"{fwd_deriv:.3f}"
        row_data['Forward Truncation Error'] = f"{fwd_trunc:.3f}"
    
    if show_central:
        cent_deriv = central_difference(f, x_point, step_size)
        cent_trunc = abs(-(step_size**2 / 6) * f_triple_prime(x_point))
        row_data['Central f\'(x)'] = f"{cent_deriv:.3f}"
        row_data['Central Truncation Error'] = f"{cent_trunc:.3f}"
    
    if show_backward:
        back_deriv = backward_difference(f, x_point, step_size)
        back_trunc = abs((step_size / 2) * f_double_prime(x_point - step_size))
        row_data['Backward f\'(x)'] = f"{back_deriv:.3f}"
        row_data['Backward Truncation Error'] = f"{back_trunc:.3f}"
    
    # Add timestamp for ordering
    row_data['Timestamp'] = pd.Timestamp.now()
    
    # Append new data to history
    st.session_state.table_data.append(row_data)
    st.session_state.previous_h = step_size

# Create DataFrame from history
if st.session_state.table_data:
    df = pd.DataFrame(st.session_state.table_data)
    
    # Sort by timestamp in descending order and drop timestamp column
    df = df.sort_values('Timestamp', ascending=True)
    df = df.drop('Timestamp', axis=1)
    
    # Add exact derivative for reference
    st.write(f"Exact derivative at x = {x_point:.3f}: {exact_derivative:.3f}")
    
    # Display table with custom formatting
    st.dataframe(
        df,
        hide_index=True,
        use_container_width=True
    )
    
    # Add clear button for table
    if st.button("Clear Table History"):
        st.session_state.table_data = []
        st.session_state.previous_h = None
        
    
st.header("Step-by-Step Calculations")

# Create tabs for different methods
methods_enabled = [method for method, show in 
                  zip(['Forward', 'Central', 'Backward'], 
                      [show_forward, show_central, show_backward]) if show]

if methods_enabled:
    tabs = st.tabs(methods_enabled)
    
    # Custom CSS for step-by-step calculations
    st.markdown("""
        <style>
        .step-container {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .step-header {
            font-family: Kanit;
            font-weight: bold;
            color: #1f77b4;
            margin-bottom: 0.5rem;
        }
        .step-content {
            font-family: monospace;
            padding-left: 1.5rem;
            color: #333;
        }
        .calculation {
            color: #2c3e50;
            margin: 0.3rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    for tab, method in zip(tabs, methods_enabled):
        with tab:
            steps = get_step_by_step_calculation(
                method, f, x_point, step_size, 
                exact_derivative, 
                functions[selected_function]['second_derivative'],
                functions[selected_function]['third_derivative']
            )
            
            # Group steps by section
            current_section = None
            section_content = []
            
            for step in steps:
                if step.startswith(('1.', '2.', '3.', '4.', '5.')):
                    # If we have a previous section, display it
                    if current_section:
                        st.markdown(f"""
                            <div class="step-container">
                                <div class="step-header">{current_section}</div>
                                <div class="step-content">
                                    {'<br>'.join(section_content)}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        section_content = []
                    current_section = step
                else:
                    # Format calculation lines
                    formatted_step = f'<div class="calculation">{step}</div>'
                    section_content.append(formatted_step)
            
            # Display the last section
            if current_section:
                st.markdown(f"""
                    <div class="step-container">
                        <div class="step-header">{current_section}</div>
                        <div class="step-content">
                            {'<br>'.join(section_content)}
                        </div>
                    </div>
                """, unsafe_allow_html=True)
else:
    st.warning("Please select at least one method to see step-by-step calculations.")