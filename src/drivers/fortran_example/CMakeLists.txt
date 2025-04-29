function(add_example name source)
    add_executable(${name} ${source})
    target_link_libraries(${name} ${PYTHON_LIBRARIES} callpy)
endfunction()

add_example(setget_array setget_array.f90)
add_example(call_stencil call_stencil.f90)
