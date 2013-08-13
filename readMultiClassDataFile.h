#include <string>
#include <vector>

void readMultiClassDataFile(const std::string sampleDataFilename,
                            std::vector< std::vector<double> >& sampleFeatures,
                            std::vector<int>& sampleLabels);
