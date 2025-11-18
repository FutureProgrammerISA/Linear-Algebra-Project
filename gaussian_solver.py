import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import numpy as np


class GaussianEliminationSolver:
    def __init__(self, root):
        self.root = root
        self.root.title("Gaussian Elimination Solver")
        self.root.geometry("900x700")
        self.num_equations = tk.IntVar(value=3)
        self.entry_widgets = []
        self.create_widgets()

    def create_widgets(self):
        # Top frame
        top_frame = ttk.Frame(self.root, padding="10")
        top_frame.pack(fill=tk.X)
        ttk.Label(top_frame, text="Number of Equations:", font=("Arial", 12)).pack(side=tk.LEFT, padx=5)
        ttk.Spinbox(top_frame, from_=2, to=10, textvariable=self.num_equations, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_frame, text="Generate Matrix", command=self.generate_matrix_inputs).pack(side=tk.LEFT, padx=10)

        # Matrix input frame
        self.matrix_frame = ttk.LabelFrame(self.root, text="Augmented Matrix Input", padding="10")
        self.matrix_frame.pack(fill=tk.BOTH, padx=10, pady=5)

        # Buttons
        button_frame = ttk.Frame(self.root, padding="10")
        button_frame.pack(fill=tk.X)
        ttk.Button(button_frame, text="Solve with Steps", command=self.solve_system).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Output", command=lambda: self.output_text.delete(1.0, tk.END)).pack(
            side=tk.LEFT, padx=5)

        # Output
        output_frame = ttk.LabelFrame(self.root, text="Solution Steps", padding="10")
        output_frame.pack(fill=tk.BOTH, padx=10, pady=5, expand=True)
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, font=("Courier", 10), height=20)
        self.output_text.pack(fill=tk.BOTH, expand=True)

        self.generate_matrix_inputs()

    def generate_matrix_inputs(self):
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()

        self.entry_widgets = []
        n = self.num_equations.get()

        # Validate n is reasonable
        if n < 2 or n > 10:
            messagebox.showerror("Error", "Number of equations must be between 2 and 10")
            self.num_equations.set(3)
            n = 3

        ttk.Label(self.matrix_frame, text="Enter coefficients and constants:", font=("Arial", 10, "bold")).grid(row=0,
                                                                                                                column=0,
                                                                                                                columnspan=n + 2,
                                                                                                                pady=5)

        # Headers
        for j in range(n):
            ttk.Label(self.matrix_frame, text=f"x{j + 1}", font=("Arial", 9)).grid(row=1, column=j, padx=5)
        ttk.Label(self.matrix_frame, text="|", font=("Arial", 9)).grid(row=1, column=n, padx=2)
        ttk.Label(self.matrix_frame, text="b", font=("Arial", 9)).grid(row=1, column=n + 1, padx=5)

        # Entry fields
        for i in range(n):
            row_entries = []
            for j in range(n + 1):
                entry = ttk.Entry(self.matrix_frame, width=8, font=("Arial", 10))
                entry.grid(row=i + 2, column=j if j < n else j + 1, padx=5, pady=5)
                entry.insert(0, "0")
                row_entries.append(entry)
            self.entry_widgets.append(row_entries)

    def get_matrix(self):
        n = self.num_equations.get()

        # Check if entry_widgets is properly initialized
        if not self.entry_widgets or len(self.entry_widgets) != n:
            messagebox.showerror("Error", "Please generate matrix first!")
            return None

        try:
            matrix = []
            for i in range(n):
                row = []
                for j in range(n + 1):
                    value_str = self.entry_widgets[i][j].get().strip()
                    if not value_str:
                        messagebox.showerror("Input Error", f"Empty value at row {i + 1}, column {j + 1}")
                        return None
                    row.append(float(value_str))
                matrix.append(row)
            return np.array(matrix, dtype=float)
        except ValueError as e:
            messagebox.showerror("Input Error", f"Please enter valid numbers only!\nError: {str(e)}")
            return None
        except IndexError as e:
            messagebox.showerror("Error", "Matrix structure error. Please regenerate the matrix.")
            return None

    def log(self, text):
        try:
            self.output_text.insert(tk.END, text)
            self.output_text.see(tk.END)
        except:
            pass  # Fail silently if output widget has issues

    def print_matrix(self, matrix, desc):
        if matrix is None or matrix.size == 0:
            return

        self.log(f"\n{desc}\n" + "-" * 60 + "\n")
        n = matrix.shape[0]
        for i in range(n):
            self.log("[ " + " ".join(f"{matrix[i][j]:8.3f}" for j in range(n)) + f" | {matrix[i][n]:8.3f} ]\n")
        self.log("\n")

    def gaussian_elimination(self, matrix):
        if matrix is None or matrix.size == 0:
            self.log("Error: Invalid matrix\n")
            return None

        n = matrix.shape[0]

        # Check if matrix is properly shaped
        if matrix.shape[1] != n + 1:
            self.log("Error: Matrix dimensions are incorrect\n")
            return None

        A = matrix.copy()

        self.print_matrix(A, "Initial Augmented Matrix:")
        self.log("=" * 60 + "\nFORWARD ELIMINATION PHASE\n" + "=" * 60 + "\n\n")

        # Forward elimination
        for i in range(n):
            # Partial pivoting - find row with largest absolute value in column i
            max_row = i
            max_val = abs(A[i][i])
            for k in range(i + 1, n):
                if abs(A[k][i]) > max_val:
                    max_val = abs(A[k][i])
                    max_row = k

            # Swap rows if needed
            if max_row != i:
                A[[i, max_row]] = A[[max_row, i]]
                self.log(f"Step: Swapped Row {i + 1} with Row {max_row + 1} (partial pivoting)\n")
                self.print_matrix(A, f"After swapping rows {i + 1} and {max_row + 1}:")

            # Check for zero or near-zero pivot
            if abs(A[i][i]) < 1e-10:
                self.log("\nError: System has no unique solution (zero or near-zero pivot encountered)!\n")
                self.log("The system may be inconsistent or have infinitely many solutions.\n")
                return None

            # Eliminate below pivot
            for k in range(i + 1, n):
                if abs(A[k][i]) > 1e-10:  # Only eliminate if coefficient is significant
                    factor = A[k][i] / A[i][i]
                    self.log(f"Step: Eliminate x{i + 1} from Row {k + 1} using Row {i + 1}\n")
                    self.log(
                        f"      Factor = {factor:.3f}, Operation: R{k + 1} = R{k + 1} - ({factor:.3f}) × R{i + 1}\n")
                    A[k] = A[k] - factor * A[i]
                    self.print_matrix(A, f"After eliminating from Row {k + 1}:")

        # Back substitution
        self.log("=" * 60 + "\nBACK SUBSTITUTION PHASE\n" + "=" * 60 + "\n\n")
        x = np.zeros(n)

        for i in range(n - 1, -1, -1):
            # Check for zero diagonal element
            if abs(A[i][i]) < 1e-10:
                self.log(f"\nError: Cannot solve - zero diagonal at position {i + 1}\n")
                return None

            # Calculate x[i]
            sum_val = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x[i] = (A[i][n] - sum_val) / A[i][i]

            # Check for numerical overflow or invalid results
            if not np.isfinite(x[i]):
                self.log(f"\nError: Numerical instability detected at x{i + 1}\n")
                return None

            self.log(f"Step: Solving for x{i + 1}\n      x{i + 1} = ")
            if i == n - 1:
                self.log(f"{A[i][n]:.3f} / {A[i][i]:.3f} = {x[i]:.6f}\n\n")
            else:
                self.log(f"({A[i][n]:.3f}" + "".join(
                    f" - {A[i][j]:.3f}×{x[j]:.3f}" for j in range(i + 1, n)) + f") / {A[i][i]:.3f} = {x[i]:.6f}\n\n")

        return x

    def solve_system(self):
        try:
            self.output_text.delete(1.0, tk.END)
            matrix = self.get_matrix()
            if matrix is None:
                return

            self.log("GAUSSIAN ELIMINATION SOLVER\n" + "=" * 60 + "\n\n")
            solution = self.gaussian_elimination(matrix)

            if solution is not None:
                self.log("=" * 60 + "\nFINAL SOLUTION\n" + "=" * 60 + "\n\n")
                for i, val in enumerate(solution):
                    self.log(f"x{i + 1} = {val:.6f}\n")

                # Verification
                self.log("\n" + "=" * 60 + "\nVERIFICATION\n" + "=" * 60 + "\n\n")
                n = matrix.shape[0]
                for i in range(n):
                    result = sum(matrix[i][j] * solution[j] for j in range(n))
                    error = abs(result - matrix[i][n])
                    self.log(f"Equation {i + 1}: {result:.6f} ≈ {matrix[i][n]:.6f} (Error: {error:.2e})\n")
        except Exception as e:
            messagebox.showerror("Unexpected Error", f"An error occurred:\n{str(e)}")
            self.log(f"\nUnexpected error: {str(e)}\n")


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = GaussianEliminationSolver(root)
        root.mainloop()
    except Exception as e:
        print(f"Failed to start application: {e}")
        messagebox.showerror("Startup Error", f"Failed to start application:\n{str(e)}")