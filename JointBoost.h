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

#pragma once

#include <string>
#include <vector>

class JointBoost {
public:
    JointBoost() : sampleTotal_(0), featureTotal_(0), classTotal_(0) {}
    
    void setTrainingSamples(const std::string& trainingDataFilename);
    void train(const int roundTotal, const bool verbose = false);
    
    double predict(const int classIndex, const std::vector<double>& featureVector) const;
    
    int classTotal() const { return classTotal_; }
    
    void writeFile(const std::string filename) const;
    void readFile(const std::string filename);
    
private:
    class SharingClassSet {
    public:
        SharingClassSet() : classSetBitCode_(0) {}
        SharingClassSet(const SharingClassSet sharingClassSet, const int newClass);
        
        void setClassSetBitCode(const int classSetBitCode) { classSetBitCode_ = classSetBitCode; }
        
        bool isEmpty() const;
        bool isContain(const int classIndex) const;
        int getClassSetBitCode() const { return classSetBitCode_; }
        
    private:
        int classSetBitCode_;
    };
    
    class SharingDecisionStump {
    public:
        SharingDecisionStump() : featureIndex_(-1), error_(-1) {}
        
        void initialize(const int classTotal, const SharingClassSet classSet);
        void setClassSet(const SharingClassSet classSet);
        void set(const int featureIndex,
                 const double threshold,
                 const double outpurLarger,
                 const double outputSmaller,
                 const std::vector<double>& classSpecificConstants,
                 const double error);
        void set(const int classSetBitCode,
                 const int featureIndex,
                 const double threshold,
                 const double outpurLarger,
                 const double outputSmaller,
                 const std::vector<double>& classSpecificConstants);
        void setError(const double error) { error_ = error; }
        
        double evaluate(const int classIndex, const double featureValue) const;
        double evaluate(const int classIndex, const std::vector<double>& featureVector) const;
        
        SharingClassSet classSet() const { return classSet_; }
        int featureIndex() const { return featureIndex_; }
        double threshold() const { return threshold_; }
        double outputLarger() const { return outputLarger_; }
        double outputSmaller() const { return outputSmaller_; }
        std::vector<double> classSpecificConstants() const { return classSpecificConstants_; }
        double error() const { return error_; }
        
    private:
        SharingClassSet classSet_;
        int featureIndex_;
        double threshold_;
        double outputLarger_;
        double outputSmaller_;
        std::vector<double> classSpecificConstants_;
        double error_;
    };
    
    void initializeWeights();
    void sortSampleIndices();
    void trainRound();
    void calcWeightSums();
    void learnOptimalSharingClassifier(const int featureIndex, SharingDecisionStump& optimalSharingClassifier);
    void learnOptimalClassSetClassifier(const int featureIndex,
                                        const SharingClassSet classSet,
                                        SharingDecisionStump& optimalClassifier);
    void updateWeight(const SharingDecisionStump& optimalClassifier);
    
    
    int classTotal_;
    int featureTotal_;
    std::vector<SharingDecisionStump> weakClassifiers_;
    
    // Training samples
    int sampleTotal_;
    std::vector< std::vector<double> > samples_;
    std::vector<int> labels_;
    std::vector< std::vector<double> > weights_;
    
    // Data for training
    std::vector< std::vector<int> > sortedSampleIndices_;
    std::vector<double> weightSums_;
    std::vector<double> weightLabelSums_;
    std::vector<double> positiveWeightSums_;
    std::vector<double> negativeWeightSums_;
    std::vector<double> classSpecificConstants_;
};
