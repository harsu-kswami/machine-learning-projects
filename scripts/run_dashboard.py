# #!/usr/bin/env python3
# """
# Dashboard Runner - Launch interactive dashboards for positional encoding visualization
# """

# import sys
# import os
# import argparse
# import subprocess
# from pathlib import Path

# # Project paths
# project_root = Path(__file__).parent.parent.resolve()
# src_path = project_root / 'src'
# sys.path.insert(0, str(project_root))
# sys.path.insert(0, str(src_path))

# # ---------------- StreamlitApp Class ----------------
# class StreamlitApp:
#     def __init__(self):
#         """Initialize Streamlit app state"""
#         import streamlit as st
#         self.st = st
#         if 'attention_data' not in st.session_state:
#             st.session_state.attention_data = None
#         if 'head_analysis' not in st.session_state:
#             st.session_state.head_analysis = None

#     # ---------------- Add this method ----------------
#     def _analyze_head_specialization(self, attention_data):
#         """
#         Analyze attention head specialization.
#         Replace this stub with your actual analysis logic.
#         """
#         if attention_data is None:
#             return None

#         # Example: compute a simple summary
#         head_analysis = {
#             "num_heads": len(attention_data),
#             "max_attention": max([a.max() for a in attention_data]),
#             "min_attention": min([a.min() for a in attention_data]),
#             "mean_attention": sum([a.mean() for a in attention_data]) / len(attention_data)
#         }
#         return head_analysis

#     # ---------------- Tabs -----------------
#     def attention_patterns_tab(self):
#         st = self.st
#         st.header("Attention Patterns")
#         if st.session_state.attention_data is None:
#             st.info("No attention data available yet.")
#             return

#         # Compute head analysis
#         st.session_state.head_analysis = self._analyze_head_specialization(st.session_state.attention_data)

#         if st.session_state.head_analysis:
#             st.subheader("Head Specialization Summary")
#             st.write(st.session_state.head_analysis)
#         else:
#             st.warning("Failed to compute head specialization.")

#     def positional_encoding_tab(self):
#         st = self.st
#         st.header("Positional Encodings")
#         st.write("Visualizations for positional encodings go here.")
#         # Add your plotting code

#     # ---------------- Main Run -----------------
#     def run(self):
#         st = self.st
#         st.sidebar.title("Navigation")
#         tab = st.sidebar.radio("Select Tab", ["Attention Patterns", "Positional Encodings"])
#         if tab == "Attention Patterns":
#             self.attention_patterns_tab()
#         elif tab == "Positional Encodings":
#             self.positional_encoding_tab()


# # ----------------- Streamlit Launcher ----------------
# def find_streamlit_app():
#     """Locate Streamlit app file"""
#     candidates = [
#         project_root / "src" / "interactive" / "streamlit_app.py",
#         project_root / "interactive" / "streamlit_app.py",
#         project_root / "scripts" / "streamlit_app.py",
#         project_root / "interactive" / "streamlit_app" / "run_streamlit_app.py"
#     ]
#     for path in candidates:
#         if path.exists():
#             return path
#     return None

# def launch_streamlit_dashboard(port=8501, host='localhost'):
#     """Launch Streamlit dashboard using subprocess"""
#     print("Launching Streamlit Dashboard")
#     print(f"   Host: {host}")
#     print(f"   Port: {port}")
#     print(f"   URL: http://{host}:{port}")

#     app_path = find_streamlit_app()
#     if app_path is None:
#         print("Failed to launch Streamlit: Streamlit app file not found.")
#         print("Expected locations:")
#         print("  - interactive/streamlit_app.py")
#         print("  - interactive/streamlit_app/run_streamlit_app.py")
#         print("  - scripts/streamlit_app.py")
#         return

#     try:
#         subprocess.run([
#             sys.executable, "-m", "streamlit", "run", str(app_path),
#             "--server.port", str(port),
#             "--server.address", host
#         ], check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"Failed to launch Streamlit: {e}")

# # ---------------- Gradio Launcher ----------------
# def launch_gradio_interface(port=7860, host='localhost', share=False):
#     """Launch Gradio interface"""
#     print("Launching Gradio Interface")
#     print(f"   Host: {host}")
#     print(f"   Port: {port}")
#     print(f"   Share: {share}")
#     try:
#         from interactive.gradio_interface import create_gradio_interface
#         interface = create_gradio_interface()
#         interface.launch(server_name=host, server_port=port, share=share, debug=True)
#     except ImportError as e:
#         print(f"Failed to import Gradio interface: {e}")
#     except Exception as e:
#         print(f"Failed to launch Gradio: {e}")

# # ---------------- Plotly Dash Launcher ----------------
# def launch_plotly_dashboard(port=8050, host='localhost', debug=False):
#     """Launch Plotly Dash dashboard"""
#     print("Launching Plotly Dash Dashboard")
#     print(f"   Host: {host}")
#     print(f"   Port: {port}")
#     print(f"   Debug: {debug}")
#     try:
#         from interactive.plotly_dashboard import create_plotly_dashboard
#         dashboard = create_plotly_dashboard(debug=debug)
#         dashboard.run_server(host=host, port=port, debug=debug)
#     except ImportError as e:
#         print(f"Failed to import Plotly dashboard: {e}")
#     except Exception as e:
#         print(f"Failed to launch Plotly dashboard: {e}")

# # ---------------- Jupyter Launcher ----------------
# def launch_jupyter_interface():
#     """Launch Jupyter notebook interface"""
#     print("Launching Jupyter Interface")
#     try:
#         from interactive.widget_components import launch_notebook_interface
#         from IPython import get_ipython
#         if get_ipython() is not None:
#             launch_notebook_interface()
#             print("Jupyter interface launched successfully")
#         else:
#             print("Not running in Jupyter environment. Run this in a Jupyter notebook.")
#     except ImportError as e:
#         print(f"Failed to import Jupyter interface: {e}")

# # ---------------- Dependency Check ----------------
# def check_dependencies():
#     """Check required packages"""
#     print("Checking Dependencies")
#     dependencies = {
#         'streamlit': ['streamlit'],
#         'gradio': ['gradio'],
#         'plotly_dash': ['dash', 'plotly'],
#         'jupyter': ['ipywidgets', 'jupyter']
#     }
#     available = []
#     for interface, pkgs in dependencies.items():
#         all_available = True
#         for pkg in pkgs:
#             try:
#                 __import__(pkg)
#             except ImportError:
#                 all_available = False
#                 break
#         if all_available:
#             available.append(interface)
#             print(f"  {interface}: Available")
#         else:
#             print(f"  {interface}: Missing packages: {pkgs}")
#     print(f"Available Interfaces: {len(available)}/4")
#     return available

# # ---------------- Help ----------------
# def show_help():
#     help_text = """
# Positional Encoding Visualizer - Dashboard Launcher

# USAGE:
#     python run_dashboard.py [interface] [options]

# INTERFACES:
#     streamlit    Launch Streamlit dashboard (default)
#     gradio       Launch Gradio interface
#     plotly       Launch Plotly Dash dashboard
#     jupyter      Launch Jupyter notebook interface
#     check        Check available interfaces

# OPTIONS:
#     --port PORT     Set port number
#     --host HOST     Set host address (default: localhost)
#     --share         Enable public sharing (Gradio only)
#     --debug         Enable debug mode (Plotly only)
#     --help          Show this help message

# EXAMPLES:
#     python run_dashboard.py streamlit --port 8501
#     python run_dashboard.py gradio --share
#     python run_dashboard.py plotly --debug
#     python run_dashboard.py check
# """
#     print(help_text)

# # ---------------- Main ----------------
# def main():
#     parser = argparse.ArgumentParser(description="Launch dashboards",
#                                      formatter_class=argparse.RawDescriptionHelpFormatter)
#     parser.add_argument(
#         "interface", nargs="?", default="streamlit",
#         choices=["streamlit", "gradio", "plotly", "jupyter", "check", "help"]
#     )
#     parser.add_argument("--port", type=int)
#     parser.add_argument("--host", type=str, default="localhost")
#     parser.add_argument("--share", action="store_true")
#     parser.add_argument("--debug", action="store_true")
#     args = parser.parse_args()

#     if args.interface == "help":
#         show_help()
#         return

#     if args.interface == "check":
#         check_dependencies()
#         return

#     default_ports = {
#         'streamlit': 8501,
#         'gradio': 7860,
#         'plotly': 8050,
#         'jupyter': None
#     }
#     port = args.port or default_ports.get(args.interface)

#     print("Positional Encoding Visualizer")
#     print("="*50)

#     try:
#         if args.interface == "streamlit":
#             launch_streamlit_dashboard(port=port, host=args.host)
#         elif args.interface == "gradio":
#             launch_gradio_interface(port=port, host=args.host, share=args.share)
#         elif args.interface == "plotly":
#             launch_plotly_dashboard(port=port, host=args.host, debug=args.debug)
#         elif args.interface == "jupyter":
#             launch_jupyter_interface()
#     except KeyboardInterrupt:
#         print("\nDashboard stopped by user")
#     except Exception as e:
#         print(f"Failed to launch dashboard: {e}")


# if __name__ == "__main__":
#     main()
# #!/usr/bin/env python3
# """
# Dashboard Runner - Launch interactive dashboards for positional encoding visualization
# """

# import sys
# import os
# import argparse
# import subprocess
# from pathlib import Path

# # Project paths
# project_root = Path(__file__).parent.parent.resolve()
# src_path = project_root / 'src'
# sys.path.insert(0, str(project_root))
# sys.path.insert(0, str(src_path))

# # ---------------- StreamlitApp Class ----------------
# class StreamlitApp:
#     def __init__(self):
#         """Initialize Streamlit app state"""
#         import streamlit as st
#         self.st = st
#         if 'attention_data' not in st.session_state:
#             st.session_state.attention_data = None
#         if 'head_analysis' not in st.session_state:
#             st.session_state.head_analysis = None

#     # ---------------- Add this method ----------------
#     def _analyze_head_specialization(self, attention_data):
#         """
#         Analyze attention head specialization.
#         Replace this stub with your actual analysis logic.
#         """
#         if attention_data is None:
#             return None

#         # Example: compute a simple summary
#         head_analysis = {
#             "num_heads": len(attention_data),
#             "max_attention": max([a.max() for a in attention_data]),
#             "min_attention": min([a.min() for a in attention_data]),
#             "mean_attention": sum([a.mean() for a in attention_data]) / len(attention_data)
#         }
#         return head_analysis

#     # ---------------- Tabs -----------------
#     def attention_patterns_tab(self):
#         st = self.st
#         st.header("Attention Patterns")
#         if st.session_state.attention_data is None:
#             st.info("No attention data available yet.")
#             return

#         # Compute head analysis
#         st.session_state.head_analysis = self._analyze_head_specialization(st.session_state.attention_data)

#         if st.session_state.head_analysis:
#             st.subheader("Head Specialization Summary")
#             st.write(st.session_state.head_analysis)
#         else:
#             st.warning("Failed to compute head specialization.")

#     def positional_encoding_tab(self):
#         st = self.st
#         st.header("Positional Encodings")
#         st.write("Visualizations for positional encodings go here.")
#         # Add your plotting code

#     # ---------------- Main Run -----------------
#     def run(self):
#         st = self.st
#         st.sidebar.title("Navigation")
#         tab = st.sidebar.radio("Select Tab", ["Attention Patterns", "Positional Encodings"])
#         if tab == "Attention Patterns":
#             self.attention_patterns_tab()
#         elif tab == "Positional Encodings":
#             self.positional_encoding_tab()


# # ----------------- Streamlit Launcher ----------------
# def find_streamlit_app():
#     """Locate Streamlit app file"""
#     candidates = [
#         project_root / "src" / "interactive" / "streamlit_app.py",
#         project_root / "interactive" / "streamlit_app.py",
#         project_root / "scripts" / "streamlit_app.py",
#         project_root / "interactive" / "streamlit_app" / "run_streamlit_app.py"
#     ]
#     for path in candidates:
#         if path.exists():
#             return path
#     return None

# def launch_streamlit_dashboard(port=8501, host='localhost'):
#     """Launch Streamlit dashboard using subprocess"""
#     print("Launching Streamlit Dashboard")
#     print(f"   Host: {host}")
#     print(f"   Port: {port}")
#     print(f"   URL: http://{host}:{port}")

#     app_path = find_streamlit_app()
#     if app_path is None:
#         print("Failed to launch Streamlit: Streamlit app file not found.")
#         print("Expected locations:")
#         print("  - interactive/streamlit_app.py")
#         print("  - interactive/streamlit_app/run_streamlit_app.py")
#         print("  - scripts/streamlit_app.py")
#         return

#     try:
#         subprocess.run([
#             sys.executable, "-m", "streamlit", "run", str(app_path),
#             "--server.port", str(port),
#             "--server.address", host
#         ], check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"Failed to launch Streamlit: {e}")

# # ---------------- Gradio Launcher ----------------
# def launch_gradio_interface(port=7860, host='localhost', share=False):
#     """Launch Gradio interface"""
#     print("Launching Gradio Interface")
#     print(f"   Host: {host}")
#     print(f"   Port: {port}")
#     print(f"   Share: {share}")
#     try:
#         from interactive.gradio_interface import create_gradio_interface
#         interface = create_gradio_interface()
#         interface.launch(server_name=host, server_port=port, share=share, debug=True)
#     except ImportError as e:
#         print(f"Failed to import Gradio interface: {e}")
#     except Exception as e:
#         print(f"Failed to launch Gradio: {e}")

# # ---------------- Plotly Dash Launcher ----------------
# def launch_plotly_dashboard(port=8050, host='localhost', debug=False):
#     """Launch Plotly Dash dashboard"""
#     print("Launching Plotly Dash Dashboard")
#     print(f"   Host: {host}")
#     print(f"   Port: {port}")
#     print(f"   Debug: {debug}")
#     try:
#         from interactive.plotly_dashboard import create_plotly_dashboard
#         dashboard = create_plotly_dashboard(debug=debug)
#         dashboard.run_server(host=host, port=port, debug=debug)
#     except ImportError as e:
#         print(f"Failed to import Plotly dashboard: {e}")
#     except Exception as e:
#         print(f"Failed to launch Plotly dashboard: {e}")

# # ---------------- Jupyter Launcher ----------------
# def launch_jupyter_interface():
#     """Launch Jupyter notebook interface"""
#     print("Launching Jupyter Interface")
#     try:
#         from interactive.widget_components import launch_notebook_interface
#         from IPython import get_ipython
#         if get_ipython() is not None:
#             launch_notebook_interface()
#             print("Jupyter interface launched successfully")
#         else:
#             print("Not running in Jupyter environment. Run this in a Jupyter notebook.")
#     except ImportError as e:
#         print(f"Failed to import Jupyter interface: {e}")

# # ---------------- Dependency Check ----------------
# def check_dependencies():
#     """Check required packages"""
#     print("Checking Dependencies")
#     dependencies = {
#         'streamlit': ['streamlit'],
#         'gradio': ['gradio'],
#         'plotly_dash': ['dash', 'plotly'],
#         'jupyter': ['ipywidgets', 'jupyter']
#     }
#     available = []
#     for interface, pkgs in dependencies.items():
#         all_available = True
#         for pkg in pkgs:
#             try:
#                 __import__(pkg)
#             except ImportError:
#                 all_available = False
#                 break
#         if all_available:
#             available.append(interface)
#             print(f"  {interface}: Available")
#         else:
#             print(f"  {interface}: Missing packages: {pkgs}")
#     print(f"Available Interfaces: {len(available)}/4")
#     return available

# # ---------------- Help ----------------
# def show_help():
#     help_text = """
# Positional Encoding Visualizer - Dashboard Launcher

# USAGE:
#     python run_dashboard.py [interface] [options]

# INTERFACES:
#     streamlit    Launch Streamlit dashboard (default)
#     gradio       Launch Gradio interface
#     plotly       Launch Plotly Dash dashboard
#     jupyter      Launch Jupyter notebook interface
#     check        Check available interfaces

# OPTIONS:
#     --port PORT     Set port number
#     --host HOST     Set host address (default: localhost)
#     --share         Enable public sharing (Gradio only)
#     --debug         Enable debug mode (Plotly only)
#     --help          Show this help message

# EXAMPLES:
#     python run_dashboard.py streamlit --port 8501
#     python run_dashboard.py gradio --share
#     python run_dashboard.py plotly --debug
#     python run_dashboard.py check
# """
#     print(help_text)

# # ---------------- Main ----------------
# def main():
#     parser = argparse.ArgumentParser(description="Launch dashboards",
#                                      formatter_class=argparse.RawDescriptionHelpFormatter)
#     parser.add_argument(
#         "interface", nargs="?", default="streamlit",
#         choices=["streamlit", "gradio", "plotly", "jupyter", "check", "help"]
#     )
#     parser.add_argument("--port", type=int)
#     parser.add_argument("--host", type=str, default="localhost")
#     parser.add_argument("--share", action="store_true")
#     parser.add_argument("--debug", action="store_true")
#     args = parser.parse_args()

#     if args.interface == "help":
#         show_help()
#         return

#     if args.interface == "check":
#         check_dependencies()
#         return

#     default_ports = {
#         'streamlit': 8501,
#         'gradio': 7860,
#         'plotly': 8050,
#         'jupyter': None
#     }
#     port = args.port or default_ports.get(args.interface)

#     print("Positional Encoding Visualizer")
#     print("="*50)

#     try:
#         if args.interface == "streamlit":
#             launch_streamlit_dashboard(port=port, host=args.host)
#         elif args.interface == "gradio":
#             launch_gradio_interface(port=port, host=args.host, share=args.share)
#         elif args.interface == "plotly":
#             launch_plotly_dashboard(port=port, host=args.host, debug=args.debug)
#         elif args.interface == "jupyter":
#             launch_jupyter_interface()
#     except KeyboardInterrupt:
#         print("\nDashboard stopped by user")
#     except Exception as e:
#         print(f"Failed to launch dashboard: {e}")


# if __name__ == "__main__":
#     main()

