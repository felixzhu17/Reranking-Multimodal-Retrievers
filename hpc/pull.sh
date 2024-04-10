# Fetch the latest changes from the origin
git fetch origin
if [ $? -ne 0 ]; then
    echo "Failed to fetch changes from origin."
    return 1
fi

# Pull changes from the master branch
git pull origin master
if [ $? -ne 0 ]; then
    echo "Error occurred during git pull. Resolve conflicts or other issues before proceeding."
    return 1
fi
