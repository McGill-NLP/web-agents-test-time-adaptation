# check that SUFFIX is set, otherwise give an error
if [ -z "$SUFFIX" ]; then
    echo "SUFFIX is not set. Please set it before running this script. Stopping."
    return 1
fi

HOST="mcgill-nlp.org"

# For agentlab

export AGENTLAB_EXP_ROOT=f"/network/scratch/s/shind/agentlab_results"

# Webarena

export WA_HOMEPAGE="https://wa-homepage-${SUFFIX}.${HOST}"
export WA_SHOPPING="https://wa-shopping-${SUFFIX}.${HOST}"
export WA_SHOPPING_ADMIN="https://wa-shopping-admin-${SUFFIX}.${HOST}/admin"
export WA_REDDIT="https://wa-forum-${SUFFIX}.${HOST}"
export WA_GITLAB="https://wa-gitlab-${SUFFIX}.${HOST}"
export WA_WIKIPEDIA="https://wa-wikipedia-${SUFFIX}.${HOST}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="https://wa-openstreetmap-${SUFFIX}.${HOST}"
export WA_FULL_RESET="https://wa-reset-${SUFFIX}.${HOST}"

# visualwebarena

export VWA_HOMEPAGE="https://vwa-homepage-${SUFFIX}.${HOST}"
export VWA_SHOPPING="https://vwa-shopping-${SUFFIX}.${HOST}" # different from webarena!
export VWA_REDDIT="https://vwa-forum-${SUFFIX}.${HOST}"
export VWA_CLASSIFIEDS="https://vwa-classifieds-${SUFFIX}.${HOST}"
export VWA_WIKIPEDIA="https://vwa-wikipedia-${SUFFIX}.${HOST}" # different from webarena!  
export VWA_FULL_RESET="https://vwa-reset-${SUFFIX}.${HOST}"

export VWA_CLASSIFIEDS_RESET_TOKEN="4b61655535e7ed388f0d40a93600254c"
