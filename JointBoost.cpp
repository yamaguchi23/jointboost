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

#include "JointBoost.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include "readMultiClassDataFile.h"

struct SampleElement {
    int sampleIndex;
    double sampleValue;
    
    bool operator<(const SampleElement& comparisonElement) const { return sampleValue < comparisonElement.sampleValue; }
};


JointBoost::SharingClassSet::SharingClassSet(const SharingClassSet sharingClassSet, const int newClass) {
    classSetBitCode_ = sharingClassSet.classSetBitCode_;
    classSetBitCode_ |= (1 << newClass);
}

bool JointBoost::SharingClassSet::isEmpty() const {
    return (classSetBitCode_ == 0);
}

bool JointBoost::SharingClassSet::isContain(const int classIndex) const {
    return ((classSetBitCode_ & (1 << classIndex)) != 0);
}


void JointBoost::SharingDecisionStump::initialize(const int classTotal, const SharingClassSet classSet) {
    classSet_ = classSet;
    featureIndex_ = -1;
    error_ = -1;
    classSpecificConstants_.resize(classTotal);
}

void JointBoost::SharingDecisionStump::set(const int featureIndex,
                                           const double threshold,
                                           const double outpurLarger,
                                           const double outputSmaller,
                                           const std::vector<double>& classSpecificConstants,
                                           const double error)
{
    featureIndex_ = featureIndex;
    threshold_ = threshold;
    outputLarger_ = outpurLarger;
    outputSmaller_ = outputSmaller;
    classSpecificConstants_ = classSpecificConstants;
    error_ = error;
}

void JointBoost::SharingDecisionStump::set(const int classSetBitCode,
                                           const int featureIndex,
                                           const double threshold,
                                           const double outpurLarger,
                                           const double outputSmaller,
                                           const std::vector<double>& classSpecificConstants)
{
    classSet_.setClassSetBitCode(classSetBitCode);
    featureIndex_ = featureIndex;
    threshold_ = threshold;
    outputLarger_ = outpurLarger;
    outputSmaller_ = outputSmaller;
    classSpecificConstants_ = classSpecificConstants;
}

double JointBoost::SharingDecisionStump::evaluate(const int classIndex, const double featureValue) const {
    if (classSet_.isContain(classIndex)) {
        if (featureValue > threshold_) return outputLarger_;
        else return outputSmaller_;
    } else {
        return classSpecificConstants_[classIndex];
    }
}

double JointBoost::SharingDecisionStump::evaluate(const int classIndex, const std::vector<double>& featureVector) const {
    return evaluate(classIndex, featureVector[featureIndex_]);
}


void JointBoost::setTrainingSamples(const std::string& trainingDataFilename) {
    readMultiClassDataFile(trainingDataFilename, samples_, labels_);
    sampleTotal_ = static_cast<int>(samples_.size());
    if (sampleTotal_ == 0) {
        std::cerr << "error: no training sample" << std::endl;
        exit(1);
    }
    featureTotal_ = static_cast<int>(samples_[0].size());
    classTotal_ = 0;
    for (int sampleIndex = 0; sampleIndex < sampleTotal_; ++sampleIndex) {
        if (labels_[sampleIndex] >= classTotal_) {
            classTotal_ = labels_[sampleIndex] + 1;
        }
    }
    
    initializeWeights();
    sortSampleIndices();
    
    weakClassifiers_.clear();
    
    weightSums_.resize(classTotal_);
    weightLabelSums_.resize(classTotal_);
    positiveWeightSums_.resize(classTotal_);
    negativeWeightSums_.resize(classTotal_);
    classSpecificConstants_.resize(classTotal_);
}

void JointBoost::train(const int roundTotal, const bool verbose) {
    for (int roundCount = 0; roundCount < roundTotal; ++roundCount) {
        trainRound();
        
        if (verbose) {
            std::cout << "Round " << roundCount << ": " << std::endl;
            std::cout << "Class = {";
            SharingClassSet classifierClassSet = weakClassifiers_[roundCount].classSet();
            for (int c = 0; c < classTotal_; ++c) {
                if (classifierClassSet.isContain(c)) std::cout << " " << c;
            }
            std::cout << " }, ";
            std::cout << "feature = " << weakClassifiers_[roundCount].featureIndex() << ", ";
            std::cout << "threshold = " << weakClassifiers_[roundCount].threshold() << std::endl;
            std::cout << "output = [" << weakClassifiers_[roundCount].outputLarger() << ", ";
            std::cout << weakClassifiers_[roundCount].outputSmaller() << "], ";
            std::vector<double> classifierConstants = weakClassifiers_[roundCount].classSpecificConstants();
            std::cout << "[";
            for (int c = 0; c < classTotal_; ++c) std::cout << " " << classifierConstants[c];
            std::cout << "], ";
            std::cout << "error = " << weakClassifiers_[roundCount].error() << std::endl;
        }
    }
    
    // Prediction test
    if (verbose) {
        int correctTotal = 0;
        for (int sampleIndex = 0; sampleIndex < sampleTotal_; ++sampleIndex) {
            int maxClassIndex = -1;
            double maxScore = 0;
            for (int classIndex = 0; classIndex < classTotal_; ++classIndex) {
                double score = predict(classIndex, samples_[sampleIndex]);

                if (maxClassIndex < 0 || score > maxScore) {
                    maxClassIndex = classIndex;
                    maxScore = score;
                }
            }
            
            if (maxClassIndex == labels_[sampleIndex]) ++correctTotal;
        }
        
        std::cout << std::endl;
        std::cout << "Training set: ";
        std::cout << static_cast<double>(correctTotal)/sampleTotal_;
        std::cout << " (" << correctTotal << " / " << sampleTotal_ << ")" << std::endl;
    }
}

double JointBoost::predict(const int classIndex, const std::vector<double>& featureVector) const {
    double score = 0.0;
    for (int classifierIndex = 0; classifierIndex < static_cast<int>(weakClassifiers_.size()); ++classifierIndex) {
        score += weakClassifiers_[classifierIndex].evaluate(classIndex, featureVector);
    }
    
    return score;
}


void JointBoost::initializeWeights() {
    double initialWeight = 1.0/sampleTotal_;
    
    weights_.resize(sampleTotal_);
    for (int i = 0; i < sampleTotal_; ++i) {
        weights_[i].resize(classTotal_);
        for (int c = 0; c < classTotal_; ++c) weights_[i][c] = initialWeight;
    }
}

void JointBoost::sortSampleIndices() {
    sortedSampleIndices_.resize(featureTotal_);
    for (int featureIndex = 0; featureIndex < featureTotal_; ++featureIndex) {
        std::vector<SampleElement> featureElements(sampleTotal_);
        for (int sampleIndex = 0; sampleIndex < sampleTotal_; ++sampleIndex) {
            featureElements[sampleIndex].sampleIndex = sampleIndex;
            featureElements[sampleIndex].sampleValue = samples_[sampleIndex][featureIndex];
        }
        std::sort(featureElements.begin(), featureElements.end());
        
        sortedSampleIndices_[featureIndex].resize(sampleTotal_);
        for (int i = 0; i < sampleTotal_; ++i) {
            sortedSampleIndices_[featureIndex][i] = featureElements[i].sampleIndex;
        }
    }
}

void JointBoost::trainRound() {
    calcWeightSums();
    
    SharingDecisionStump bestClassifier;
    for (int featureIndex = 0; featureIndex < featureTotal_; ++featureIndex) {
        SharingDecisionStump optimalClassifier;
        learnOptimalSharingClassifier(featureIndex, optimalClassifier);
        
        if (bestClassifier.error() < 0 || optimalClassifier.error() < bestClassifier.error()) {
            bestClassifier = optimalClassifier;
        }
    }
    
    updateWeight(bestClassifier);
    
    weakClassifiers_.push_back(bestClassifier);
}

void JointBoost::calcWeightSums() {
    for (int classIndex = 0; classIndex < classTotal_; ++classIndex) {
        weightSums_[classIndex] = 0;
        weightLabelSums_[classIndex] = 0;
        positiveWeightSums_[classIndex] = 0;
        negativeWeightSums_[classIndex] = 0;
        for (int sampleIndex = 0; sampleIndex < sampleTotal_; ++sampleIndex) {
            weightSums_[classIndex] += weights_[sampleIndex][classIndex];
            if (labels_[sampleIndex] == classIndex) {
                weightLabelSums_[classIndex] += weights_[sampleIndex][classIndex];
                positiveWeightSums_[classIndex] += weights_[sampleIndex][classIndex];
            } else {
                weightLabelSums_[classIndex] -= weights_[sampleIndex][classIndex];
                negativeWeightSums_[classIndex] += weights_[sampleIndex][classIndex];
            }
        }
        classSpecificConstants_[classIndex] = weightLabelSums_[classIndex]/weightSums_[classIndex];
    }
}

void JointBoost::learnOptimalSharingClassifier(const int featureIndex, SharingDecisionStump& optimalSharingClassifier) {
    std::vector<SharingDecisionStump> sharingClassifiers(classTotal_ - 1);
    
    SharingClassSet sourceSet;
    for (int setClassTotal = 0; setClassTotal < classTotal_ - 1; ++setClassTotal) {
        std::vector<SharingClassSet> classSetList;
        for (int classIndex = 0; classIndex < classTotal_; ++classIndex) {
            if (sourceSet.isContain(classIndex)) continue;
            
            SharingClassSet newClassSet(sourceSet, classIndex);
            classSetList.push_back(newClassSet);
        }
        
        SharingClassSet currentBestSet;
        for (int setIndex = 0; setIndex < static_cast<int>(classSetList.size()); ++setIndex) {
            SharingDecisionStump optimalClassifier;
            learnOptimalClassSetClassifier(featureIndex, classSetList[setIndex], optimalClassifier);
            
            if (sharingClassifiers[setClassTotal].error() < 0
                || (optimalClassifier.error() > 0 && optimalClassifier.error() < sharingClassifiers[setClassTotal].error()))
            {
                sharingClassifiers[setClassTotal] = optimalClassifier;
                currentBestSet = classSetList[setIndex];
            }
        }
        
        sourceSet = currentBestSet;
    }
    
    optimalSharingClassifier = sharingClassifiers[0];
    for (int i = 0; i < classTotal_ - 1; ++i) {
        if (sharingClassifiers[i].error() < optimalSharingClassifier.error()) {
            optimalSharingClassifier = sharingClassifiers[i];
        }
    }
}

void JointBoost::learnOptimalClassSetClassifier(const int featureIndex,
                                                const SharingClassSet classSet,
                                                SharingDecisionStump& optimalClassifier)
{
    const double epsilonValue = 1e-6;
    
    optimalClassifier.initialize(classTotal_, classSet);
    
    double weightSumSubset = 0;
    double weightLabelSumSubset = 0;
    double positiveWeightSumSubset = 0;
    double negativeWeightSumSubset = 0;
    for (int classIndex = 0; classIndex < classTotal_; ++classIndex) {
        if (classSet.isContain(classIndex)) {
            weightSumSubset += weightSums_[classIndex];
            weightLabelSumSubset += weightLabelSums_[classIndex];
            positiveWeightSumSubset += positiveWeightSums_[classIndex];
            negativeWeightSumSubset += negativeWeightSums_[classIndex];
        }
    }
    
    double weightSumSubsetLarger = weightSumSubset;
    double weightLabelSumSubsetLarger = weightLabelSumSubset;
    double positiveWeightSumSubsetLarger = positiveWeightSumSubset;
    double negativeWeightSumSubsetLarger = negativeWeightSumSubset;
    
    for (int sortIndex = 0; sortIndex < sampleTotal_; ++sortIndex) {
        int sampleIndex = sortedSampleIndices_[featureIndex][sortIndex];
        double threshold = samples_[sampleIndex][featureIndex];
        for (int classIndex = 0; classIndex < classTotal_; ++classIndex) {
            if (classSet.isContain(classIndex)) {
                double sampleWeight = weights_[sampleIndex][classIndex];
                weightSumSubsetLarger -= sampleWeight;
                if (labels_[sampleIndex] == classIndex) {
                    weightLabelSumSubsetLarger -= sampleWeight;
                    positiveWeightSumSubsetLarger -= sampleWeight;
                } else {
                    weightLabelSumSubsetLarger += sampleWeight;
                    negativeWeightSumSubsetLarger -= sampleWeight;
                }
            }
        }
        
        while (sortIndex < sampleTotal_ - 1
               && samples_[sampleIndex][featureIndex] == samples_[sortedSampleIndices_[featureIndex][sortIndex + 1]][featureIndex])
        {
            ++sortIndex;
            sampleIndex = sortedSampleIndices_[featureIndex][sortIndex];
            for (int classIndex = 0; classIndex < classTotal_; ++classIndex) {
                if (classSet.isContain(classIndex)) {
                    double sampleWeight = weights_[sampleIndex][classIndex];
                    weightSumSubsetLarger -= sampleWeight;
                    if (labels_[sampleIndex] == classIndex) {
                        weightLabelSumSubsetLarger -= sampleWeight;
                        positiveWeightSumSubsetLarger -= sampleWeight;
                    } else {
                        weightLabelSumSubsetLarger += sampleWeight;
                        negativeWeightSumSubsetLarger -= sampleWeight;
                    }
                }
            }
        }
        if (sortIndex >= sampleTotal_ - 1) break;
        
        if (fabs(weightSumSubsetLarger) < epsilonValue || fabs(weightSumSubset - weightSumSubsetLarger) < epsilonValue) continue;
        
        double outputLarger = weightLabelSumSubsetLarger/weightSumSubsetLarger;
        double outputSmaller = (weightLabelSumSubset - weightLabelSumSubsetLarger)/(weightSumSubset - weightSumSubsetLarger);
        
        double error = positiveWeightSumSubsetLarger*(1.0 - outputLarger)*(1.0 - outputLarger)
            + (positiveWeightSumSubset - positiveWeightSumSubsetLarger)*(1.0 - outputSmaller)*(1.0 - outputSmaller)
            + negativeWeightSumSubsetLarger*(-1.0 - outputLarger)*(-1.0 - outputLarger)
            + (negativeWeightSumSubset - negativeWeightSumSubsetLarger)*(-1.0 - outputSmaller)*(-1.0 - outputSmaller);
        
        if (optimalClassifier.error() < 0 || error < optimalClassifier.error()) {
            double classifierThreshold = (threshold + samples_[sortedSampleIndices_[featureIndex][sortIndex + 1]][featureIndex])/2.0;

            optimalClassifier.set(featureIndex, classifierThreshold, outputLarger, outputSmaller, classSpecificConstants_, error);
        }
    }
    
    if (optimalClassifier.error() > 0) {
        double optimalError = optimalClassifier.error();
        for (int classIndex = 0; classIndex < classTotal_; ++classIndex) {
            if (!classSet.isContain(classIndex)) {
                double outputConstant = classSpecificConstants_[classIndex];
                optimalError += positiveWeightSums_[classIndex]*(1.0 - outputConstant)*(1.0 - outputConstant)
                    + negativeWeightSums_[classIndex]*(-1.0 - outputConstant)*(-1.0 - outputConstant);
            }
        }
        optimalClassifier.setError(optimalError);
    }
}

void JointBoost::updateWeight(const SharingDecisionStump& optimalClassifier) {
    int featureIndex = optimalClassifier.featureIndex();
    
    for (int sampleIndex = 0; sampleIndex < sampleTotal_; ++sampleIndex) {
        double featureValue = samples_[sampleIndex][featureIndex];
        
        for (int classIndex = 0; classIndex < classTotal_; ++classIndex) {
            double classLabel = (labels_[sampleIndex] == classIndex) ? 1.0 : -1.0;
            double confidence = optimalClassifier.evaluate(classIndex, featureValue);
            
            weights_[sampleIndex][classIndex] *= exp(-classLabel*confidence);
        }
    }
}


void JointBoost::writeFile(const std::string filename) const {
    std::ofstream outputModelStream(filename.c_str(), std::ios_base::out);
    if (outputModelStream.fail()) {
        std::cerr << "error: can't open file (" << filename << ")" << std::endl;
        exit(1);
    }
    
    int roundTotal = static_cast<int>(weakClassifiers_.size());
    outputModelStream << roundTotal << " " << classTotal_ << std::endl;
    for (int roundIndex = 0; roundIndex < roundTotal; ++roundIndex) {
        outputModelStream << weakClassifiers_[roundIndex].classSet().getClassSetBitCode() << " ";
        outputModelStream << weakClassifiers_[roundIndex].featureIndex() << " ";
        outputModelStream << weakClassifiers_[roundIndex].threshold() << " ";
        outputModelStream << weakClassifiers_[roundIndex].outputLarger() << " ";
        outputModelStream << weakClassifiers_[roundIndex].outputSmaller() << " ";
        for (int c = 0; c < classTotal_ - 1; ++c) outputModelStream << weakClassifiers_[roundIndex].classSpecificConstants()[c] << " ";
        outputModelStream << weakClassifiers_[roundIndex].classSpecificConstants()[classTotal_ - 1] << std::endl;
    }
    
    outputModelStream.close();
}

void JointBoost::readFile(const std::string filename) {
    std::ifstream inputModelStream(filename.c_str(), std::ios_base::in);
    if (inputModelStream.fail()) {
        std::cerr << "error: can't open file (" << filename << ")" << std::endl;
        exit(1);
    }
    
    int roundTotal;
    inputModelStream >> roundTotal;
    inputModelStream >> classTotal_;
    weakClassifiers_.resize(roundTotal);
    for (int roundIndex = 0; roundIndex < roundTotal; ++roundIndex) {
        int classSetBitCode;
        int featureIndex;
        double threshold, outputLarger, outputSmaller;
        std::vector<double> classSpecificConstants(classTotal_);
        inputModelStream >> classSetBitCode;
        inputModelStream >> featureIndex;
        inputModelStream >> threshold;
        inputModelStream >> outputLarger;
        inputModelStream >> outputSmaller;
        for (int c = 0; c < classTotal_; ++c) inputModelStream >> classSpecificConstants[c];
        
        weakClassifiers_[roundIndex].set(classSetBitCode, featureIndex, threshold, outputLarger, outputSmaller, classSpecificConstants);
    }
    
    inputModelStream.close();
}
