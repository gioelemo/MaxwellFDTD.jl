using Plots

function simpleWaterfall(z, offset, scale)
    # release any previous plot
    plot(scale * z[1], label="")
    
    # hold the plot
    for i = 2:size(z, 1)
        plot!(scale * z[i] + offset * (i - 1), label="")
    end
    
    # release the plot
    plot!()
end

# Example usage:
# Replace the following lines with your data
#z = randn(5, 100)  # Example data, replace with your actual data
#offset = 0.2       # Example offset, replace with your desired offset
#cale = 1     # Example scale, replace with your desired scale

#display(simpleWaterfall(z, offset, scale))