# Fetch the latest changes from the origin
git fetch origin
if [ $? -ne 0 ]; then
    echo "Failed to fetch changes from origin."
    return 1
fi

# Pull changes from the main branch
git pull origin master
if [ $? -ne 0 ]; then
    echo "Error occurred during git pull. Resolve conflicts or other issues before proceeding."
    return 1
fi

# Add all changes to the staging area
git add .
if [ $? -ne 0 ]; then
    echo "Failed to stage changes."
    return 1
fi

# Commit the changes
git commit -m "."
if [ $? -ne 0 ]; then
    echo "Failed to commit changes."
    return 1
fi

# Push the changes to the main branch
git push origin master
if [ $? -ne 0 ]; then
    echo "Failed to push changes to origin main."
    return 1
fi

echo "Git operations completed successfully."