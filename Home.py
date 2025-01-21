import streamlit as st

def main():
    st.set_page_config(
        page_title="Numerical Differentiation Theory",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown("""
        <style>
        /* Main container styling */
        .main > div {
            padding: 3rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        
        /* Header styling */
        h1, h2, h3 {
            font-family: 'Arial', sans-serif;
            color: #2c3e50;
            margin-top: 2.5rem;
            margin-bottom: 1.5rem;
            letter-spacing: -0.5px;
        }
        
        h1 {
            font-size: 2.5rem;
            text-align: center;
            padding-bottom: 1.5rem;
            border-bottom: 3px solid #e2e8f0;
        }
        
        h2 {
            font-size: 2rem;
            padding-bottom: 1rem;
            border-bottom: 2px solid #e2e8f0;
        }
        
        h3 {
            font-size: 1.5rem;
            color: #34495e;
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1rem;
            background-color: #f8fafc;
            padding: 1rem;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 3.5rem;
            background-color: white;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
            transition: all 0.3s ease;
            width: 200px; 
            text-align: center;
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #f1f5f9;
        }
        
        .stTabs [data-baseweb="tab-panel"] {
            padding: 2rem;
            background-color: white;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
            margin-top: 1rem;
        }
        
        /* Content box styling */
        .content-box {
            background-color: white;
            padding: 2rem;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            margin: 1.5rem 0;
        }
        
        .method-box {
            background-color: #f8fafc;
            padding: 1.5rem;
            border-radius: 8px;
            margin: 1rem 0;
        }
        
        .error-box {
            background: linear-gradient(to right bottom, #ffffff, #f8fafc);
            padding: 2rem;
            border-radius: 12px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            height: 100%;
        }
        
        /* Text styling */
        p, li {
            font-family: 'Arial', sans-serif;
            font-size: 1.1rem;
            line-height: 1.7;
            color: #4a5568;
        }
        
        /* Latex formula styling */
        .katex {
            font-size: 1.2rem !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("What is Numerical Differentiation?")
    with st.container():
        st.markdown("""
        <div class="content-box">
        Numerical differentiation is a technique used to approximate the derivative of a function 
        using discrete data points. While analytical differentiation gives us exact derivatives, 
        numerical methods are essential when:
        
        * Working with experimental data where we only have discrete points
        * Dealing with functions that are difficult or impossible to differentiate analytically
        * Implementing computer algorithms that need to calculate derivatives
        </div>
        """, unsafe_allow_html=True)
    
    st.header("Numerical Differentiation Methods")
    
    tabs = st.tabs(["Forward Difference", "Backward Difference", "Central Difference"])
    
    with tabs[0]:
        st.markdown("""
        <div class="method-box">
        <h3>Forward Difference Method</h3>
        The forward difference formula approximates the derivative using a point x and a small step size h:
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r"""f'(x) \approx \frac{f(x + h) - f(x)}{h}""")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="method-box">
        * First-order accuracy O(h) <br>
        * Truncation error:
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r"""E_t(h) \approx -\frac{h}{2}f''(\Theta) """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.latex(r"""for\ \ x \leq \Theta \leq x+h""")
        st.markdown("""
        <div class="method-box">
        * Good for computing derivatives at the beginning of intervals<br>
        * It is also known as two-point formula
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[1]:
        st.markdown("""
        <div class="method-box">
        <h3>Backward Difference Method</h3>
        The backward difference formula approximates the derivative using a point x and a small step size h 
        in the opposite direction:
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r"""f'(x) \approx \frac{f(x) - f(x - h)}{h}""")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="method-box">
        * First-order accuracy O(h) <br>
        * Truncation error:
        </div>
        """, unsafe_allow_html=True)
        
        st.latex(r"""E_t(h) \approx \frac{h}{2}f''(\Theta)""")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.latex(r"""for\ \ x-h \leq \Theta \leq x""")
        st.markdown("""
        <div class="method-box">           
        * Good for computing derivatives at the end of intervals
        </div>
        """, unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("""
        <div class="method-box">
        <h3>Central Difference Method</h3>
        The central difference formula provides a more accurate approximation considering
        point on both sides of x:
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="formula-box">', unsafe_allow_html=True)
        st.latex(r"""f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}""")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="method-box">
        * Second-order accuracy O(h²) <br>
        * Truncation error:
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="formula-box">', unsafe_allow_html=True)
        st.latex(r"""E_t(h) \approx -\frac{h^2}{6}f'''(\Theta) """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.latex(r"""for\ \ x-h \leq \Theta \leq x+h""")
        st.markdown("""
        <div class="method-box">            
        * Generally more accurate than forward or backward differences<br>
        * It is also known as three-point formula
        </div>
        """, unsafe_allow_html=True)
    
    st.header("Understanding Error")
    
    error_col1, error_col2 = st.columns(2)
    
    with error_col1:
        st.markdown("""
            <div class="error-box">
                <h3>1. Truncation Error</h3>
                <p>Truncation error is the error caused by approximating a mathematical procedure using a finite
                number of terms in a series expansion, rather than the exact formula.
                It arises from using finite differences to approximate derivatives. 
                The truncation error is in the order of 'h'.</p>
                <p>Truncation error shows how much accuracy is lost because we are using a formula instead of the exact 
                derivative. Smaller step sizes h reduce this error, and central differences usually have lower 
                truncation error than forward or backward differences.</p>
            </div>
        """, unsafe_allow_html=True)
    
    with error_col2:
        st.markdown("""
            <div class="error-box">
                <h3>2. Absolute Error</h3>
                <p>Absolute error is the difference between the true(exact) value and the approximated value. 
                It quantifies how much an approximation deviates from the actual value.
                It measure the accuracy of numerical computations and helps compare different numerical methods.</p>
                <p>Absolute error basically tells you how far your approximation is from the true derivative. A smaller absolute error means a more accurate result. 
                Try adjusting h to see how the error changes!</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
