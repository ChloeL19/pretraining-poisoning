commit plotting code into the codebase -- it should load in a piece of data and produce that exact plot

my (opinionated) choice for plots is altair / vega , which just saves the data used for the plot together with a spec to generate that plot as a JSON, and you can even save that JSON in the repo

save outputs neatly and close to the script that generates them. maximize neatness and clarity.

I think it also helps a lot to make plots easy to read for someone without context. It is generally high impact to spend some extra time on plots making sure they convey the information you want quickly. Add text, labels, arrows or whatever needed to help the reader.