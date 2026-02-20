#ifndef FACTORY_WEIGHTED_JACCARD_H
#define FACTORY_WEIGHTED_JACCARD_H

#define SPACE_WEIGHTED_JACCARD  "WeightedJaccard"

#include <space/space_vector_gen.h>

namespace similarity {

/*
 * Creating functions.
 */

template <typename dist_t>
struct WeightedJaccardDistance {
    
    /*
    * Important: the function is const and arguments are const as well!!!
    */
    float operator()(const dist_t* x, const dist_t* y, const size_t qty) const {
        float minSum = 0;
        float maxSum = 0;

        for (size_t i = 0; i < qty; ++i) {
            minSum += std::min(x[i], y[i]);
            maxSum += std::max(x[i], y[i]);
        }

        if (maxSum == 0) {
            return 0; //avoid division by zero
        }

        return 1 - (minSum / maxSum);
    }
};

template <typename dist_t>
Space<dist_t>* CreateSpaceWeightedJaccard(const AnyParams& AllParams) {
  return new VectorSpaceGen<dist_t, WeightedJaccardDistance<dist_t>>();
}
/*
 * End of creating functions.
 */

}

#endif