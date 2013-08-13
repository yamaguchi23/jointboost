/*
Copyright (c) 2013, Koichiro Yamaguchi
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the copyright holder nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include "readMultiClassDataFile.h"
#include "JointBoost.h"

struct ParameterJointPredict {
    bool verbose;
    std::string testDataFilename;
    std::string modelFilename;
    bool outputScoreFile;
    std::string outputScoreFilename;
};

// Prototype declaration
void exitWithUsage();
ParameterJointPredict parseCommandline(int argc, char* argv[]);

void exitWithUsage() {
    std::cerr << "usage: jointtrain [options] test_set_file model_file" << std::endl;
    std::cerr << "options:" << std::endl;
    std::cerr << "   -o: output score file" << std::endl;
    std::cerr << "   -v: verbose" << std::endl;
    
    exit(1);
}

ParameterJointPredict parseCommandline(int argc, char* argv[]) {
    ParameterJointPredict parameters;
    parameters.verbose = false;
    parameters.outputScoreFile = false;
    parameters.outputScoreFilename = "";
    
    // Options
    int argIndex;
    for (argIndex = 1; argIndex < argc; ++argIndex) {
        if (argv[argIndex][0] != '-') break;
        
        switch (argv[argIndex][1]) {
            case 'v':
                parameters.verbose = true;
                break;
            case 'o':
            {
                ++argIndex;
                if (argIndex >= argc) exitWithUsage();
                parameters.outputScoreFile = true;
                parameters.outputScoreFilename = argv[argIndex];
                break;
            }
            default:
                std::cerr << "error: undefined option" << std::endl;
                exitWithUsage();
                break;
        }
    }
    
    // Test data file
    if (argIndex >= argc) exitWithUsage();
    parameters.testDataFilename = argv[argIndex];
    
    // Model file
    ++argIndex;
    if (argIndex >= argc) exitWithUsage();
    parameters.modelFilename = argv[argIndex];
    
    return parameters;
}

int main(int argc, char* argv[]) {
    ParameterJointPredict parameters = parseCommandline(argc, argv);
    
    if (parameters.verbose) {
        std::cerr << std::endl;
        std::cerr << "Test data: " << parameters.testDataFilename << std::endl;
        std::cerr << "Model:     " << parameters.modelFilename << std::endl;
        if (parameters.outputScoreFile) {
            std::cerr << "Output score: " << parameters.outputScoreFilename << std::endl;
        }
        std::cerr << std::endl;
    }
    
    JointBoost jointBoost;
    jointBoost.readFile(parameters.modelFilename);
    int classTotal = jointBoost.classTotal();

    std::vector< std::vector<double> > testSamples;
    std::vector<int> testLabels;
    readMultiClassDataFile(parameters.testDataFilename, testSamples, testLabels);
    int testSampleTotal = static_cast<int>(testSamples.size());
    
    std::ofstream outputScoreStream;
    if (parameters.outputScoreFile) {
        outputScoreStream.open(parameters.outputScoreFilename.c_str(), std::ios_base::out);
        if (outputScoreStream.fail()) {
            std::cerr << "error: can't open file (" << parameters.outputScoreFilename << ")" << std::endl;
            exit(1);
        }
    }
    
    int correctTotal = 0;
    for (int sampleIndex = 0; sampleIndex < testSampleTotal; ++sampleIndex) {
        int maxClassIndex = -1;
        double maxScore = 0;
        for (int classIndex = 0; classIndex < classTotal; ++classIndex) {
            double score = jointBoost.predict(classIndex, testSamples[sampleIndex]);
            
            if (maxClassIndex < 0 || score > maxScore) {
                maxClassIndex = classIndex;
                maxScore = score;
            }
            
            if (parameters.outputScoreFile) outputScoreStream << score << " ";
        }
        if (parameters.outputScoreFile) outputScoreStream << std::endl;
        
        if (maxClassIndex == testLabels[sampleIndex]) ++correctTotal;
    }
    if (parameters.outputScoreFile) {
        outputScoreStream.close();
    }
    
    std::cout << "Accuracy = " << static_cast<double>(correctTotal)/testSampleTotal;
    std::cout << " (" << correctTotal << " / " << testSampleTotal << ")" << std::endl;
}
