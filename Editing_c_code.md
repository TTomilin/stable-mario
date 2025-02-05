PREREQUISITES:
1. Let 'stable-mario' denote the root folder of the repository
STEPS:
1. Run 'sudo apt install libbz2-dev'.
2. Inside 'stable-mario/CMakeLists.txt', replace the if-statement:
    
            if (LIBZIP_FOUND AND NOT LIBZIP_VERSION VERSION_LESS 1.0.0)
                include_directories(#{LIBZIP_INCLUDE_DIRS})
                link_directories(${LIBZIP_LIBRARY_DIRS})
            else()
                set(LIBZIP_LIBRARIES zip)
                add_subdirectory(third-party/libzip)
                include_directories(third-party/libzip third-party/libzip/lib)
    
    by just the else-clause:
    
            set(LIBZIP_LIBRARIES zip)
            add_subdirectory(third-party/libzip)
            include_directories(third-party/libzip third-party/libzip/lib)
    
    (See also the branch 'editable-c-code' for an instance of 'CMakeLists.txt' that works).
3. Edit the c++ code found in 'stable-mario/src' to your liking.
4. Run './update.sh' from inside 'stable-mario/src'. The c++ code used by train_model.py should now have changed.