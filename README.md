# CUDA Sobel Filter on BMP Images

This project implements a **CUDA-accelerated Sobel filter** to perform edge detection on 24-bit uncompressed BMP images. It reads an input `.bmp` file, applies the Sobel operator using GPU parallelization, and writes the output as a new `.bmp` file with the detected edges.

---

## 📌 Features

- Loads a standard 24-bit BMP image.
- Converts the image to grayscale.
- Applies the **Sobel filter** on the GPU using CUDA.
- Converts the result back to RGB and saves it as a new BMP image.
- Supports large image sizes efficiently using `dim3` CUDA grids.

---

## 🧰 Requirements

- CUDA Toolkit (e.g. CUDA 11+)
- NVIDIA GPU with CUDA support
- C/C++ compiler (e.g. `g++`)
- Supported OS: Linux, Windows (with WSL or native), macOS (with external GPU and CUDA support)

---

## 🔧 Build and Run Instructions

### 🔹 Option 1: Step-by-step

**1. Compile the code:**

```bash
nvcc proj.cu -o sobel_cuda
````

This creates the executable `sobel_cuda`.

**2. Run the program:**

```bash
./sobel_cuda input.bmp output.bmp
```

* Replace `input.bmp` with your actual image filename.
* Replace `output.bmp` with the desired output filename.

**Example:**

```bash
./sobel_cuda lena.bmp sobel_output.bmp
```

---

### 🔹 Option 2: Combined in one line

If you want to compile and run immediately:

```bash
nvcc proj.cu -o sobel_cuda && ./sobel_cuda input.bmp output.bmp
```

---

## 🖼️ Output

* The `output.bmp` image will be grayscale, with edges highlighted.
* Internally:

  * RGB is converted to grayscale.
  * Sobel kernel runs on the GPU to calculate gradient magnitude.
  * Result is capped to 255 and written back as grayscale RGB.

---

## 📂 Project Structure

```
.
├── proj.cu            # Main CUDA C source code
├── input.bmp          # Sample BMP input image (not included in repo)
├── sobel_output.bmp   # Result image (generated)
└── README.md          # Project documentation
```

---

## ⚠️ Notes

* This project only supports **24-bit BMP images** (no compression).
* Do **not** use PNG or JPEG formats unless converted beforehand.
* The program must be run on a machine with an NVIDIA GPU and the CUDA driver installed.
* BMP file row size is padded to a 4-byte boundary, which is handled in the code.

---

## 👤 Author

**Hamza Nayab**
📧 Email: [hamza.nayab48@gmail.com](mailto:hamza.nayab48@gmail.com)
GitHub: [@Hamza-Nayab](https://github.com/Hamza-Nayab)
