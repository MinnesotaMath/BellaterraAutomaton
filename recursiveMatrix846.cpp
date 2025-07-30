#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <lapack.h>
#include <lapacke.h>
#include <filesystem>
namespace fs = std::filesystem;
#define LAPACK_ROW_MAJOR 101

using namespace std;
 
class RecursiveMatrix
{
private:
    int n;
    std::vector<std::vector<double>> data;

public:
    RecursiveMatrix(int size) : n(size), data(size, std::vector<double>(size)) {}

    void set(int i, int j, double value)
    {
        data[i][j] = value;
    }

    double get(int i, int j) const
    {
        return data[i][j];
    }

    int size() const
    {
        return n;
    }
    RecursiveMatrix operator+(const RecursiveMatrix &other) const
    {
        if (this->n != other.n)
        {
            std::cerr << "Matrix sizes do not match for addition!" << std::endl;
            exit(1);
        }

        RecursiveMatrix result(n);
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                result.set(i, j, this->data[i][j] + other.data[i][j]);
            }
        }
        return result;
    }
};

std::vector<RecursiveMatrix> populateM(int k)
{
    std::vector<RecursiveMatrix> amatrices, bmatrices, cmatrices, mmatrices;

    RecursiveMatrix a1(2);
    a1.set(0, 0, 0);
    a1.set(0, 1, 1);
    a1.set(1, 0, 1);
    a1.set(1, 1, 0);
    amatrices.push_back(a1);

    RecursiveMatrix b1(2);
    b1.set(0, 0, 1);
    b1.set(0, 1, 0);
    b1.set(1, 0, 0);
    b1.set(1, 1, 1);
    bmatrices.push_back(b1);

    RecursiveMatrix c1(2);
    c1.set(0, 0, 1);
    c1.set(0, 1, 0);
    c1.set(1, 0, 0);
    c1.set(1, 1, 1);
    cmatrices.push_back(c1);
    mmatrices.push_back(a1 + b1 + c1);

    for (int i = 2; i <= k; ++i)
    {
        int matrixSize = pow(2, i);
        RecursiveMatrix ai(matrixSize), bi(matrixSize), ci(matrixSize);

        for (int rows = 0; rows < matrixSize / 2; ++rows)
        {
            for (int cols = 0; cols < matrixSize / 2; ++cols)
            {
                ai.set(rows, cols, 0);
                bi.set(rows, cols, amatrices[i - 2].get(rows, cols));
                ci.set(rows, cols, bmatrices[i - 2].get(rows, cols));
            }
        }
        for (int rows = 0; rows < matrixSize / 2; ++rows)
        {
            for (int cols = matrixSize / 2; cols < matrixSize; ++cols)
            {
                ai.set(rows, cols, cmatrices[i - 2].get(rows, cols - matrixSize / 2));
                bi.set(rows, cols, 0);
                ci.set(rows, cols, 0);
            }
        }
        for (int rows = matrixSize / 2; rows < matrixSize; ++rows)
        {
            for (int cols = 0; cols < matrixSize / 2; ++cols)
            {
                ai.set(rows, cols, cmatrices[i - 2].get(rows - matrixSize / 2, cols));
                bi.set(rows, cols, 0);
                ci.set(rows, cols, 0);
            }
        }
        for (int rows = matrixSize / 2; rows < matrixSize; ++rows)
        {
            for (int cols = matrixSize / 2; cols < matrixSize; ++cols)
            {
                ai.set(rows, cols, 0);
                bi.set(rows, cols, bmatrices[i - 2].get(rows - matrixSize / 2, cols - matrixSize / 2));
                ci.set(rows, cols, amatrices[i - 2].get(rows - matrixSize / 2, cols - matrixSize / 2));
            }
        }

        amatrices.push_back(ai);
        bmatrices.push_back(bi);
        cmatrices.push_back(ci);
        mmatrices.push_back(ai + bi + ci);
    }
    return mmatrices;
}

vector<double> findEigenvalues(RecursiveMatrix &R)
{
    int n = R.size();
    vector<double> eigenvalues(n);

    vector<double> A(n * n);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            A[i * n + j] = R.get(i, j);
        }
    }

    int info;
    vector<double> w(n);

    info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U', n, A.data(), n, w.data());

    if (info != 0)
    {
        cerr << "LAPACK eigenvalue computation failed!" << endl;
        exit(1);
    }

    return w;
}

void writeEigenvaluesToCSV(const vector<double> &eigenvalues, const string &filename)
{
    ofstream file(filename);

    if (!file)
    {
        cerr << "Error opening file: " << filename << endl;
        return;
    }

    file.precision(15);
    file << "Eigenvalues\n";
    for (double val : eigenvalues)
    {
        file << val << "\n";
    }

    file.close();
    cout << "Eigenvalues written to " << filename << endl;
}
// Added function below
void exportMatrixToCSV(const RecursiveMatrix &matrix, const std::string &filename)
{
    std::ofstream file(filename);

    if (!file)
    {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }

    int size = matrix.size();
    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < size; ++j)
        {
            file << matrix.get(i, j);
            if (j != size - 1)
            {
                file << ",";
            }
        }
        file << "\n";
    }

    file.close();
    std::cout << "Matrix saved to " << filename << std::endl;
}
// Added function below
void exportAllMatrices(const std::vector<RecursiveMatrix> &mvals)
{
    // Get the directory where the executable is located
    std::string baseDirectory = std::filesystem::current_path().string();
    std::string exportDir = baseDirectory + "/exportedMatrices";

    if (!fs::exists(exportDir))
    {
        fs::create_directory(exportDir);
    }

    for (int i = 1; i < mvals.size(); ++i)
    {
        std::string filename = exportDir + "/matrix_m" + std::to_string(i + 1) + ".csv";
        exportMatrixToCSV(mvals[i], filename);
    }
}

int main()
{
    // Moved the directory creation logic outside the loop
    // Get the current working directory
    string baseDir = std::filesystem::current_path().string();
    string outputDir = baseDir + "/eigenCSV/";

    if (!fs::exists(outputDir))
    {
        fs::create_directories(outputDir);
    }
    std::vector<RecursiveMatrix> mvals = populateM(11);
    for (int i = 0; i < 11; i++)
    {
        vector<double> eigenvalues = findEigenvalues(mvals[i]);
        writeEigenvaluesToCSV(eigenvalues, outputDir + "eigenvalues_" + to_string(i + 1) + ".csv");
    }
    exportAllMatrices(mvals);
    return 0;
}
