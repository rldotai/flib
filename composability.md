# Composing Features

Being able to compose features to create a sort of pipeline is very much desirable, but hard to implement in a fashion that is both universal and elegant.

## Specification of Features

For instance, I would like to be able to specify such a pipeline either in the code itself or from the command line.

I suspect that I haven't hit upon the right abstraction yet, but my current thoughts are that for this to work it will entail rewriting the code so that every feature function's datatype, input shape, and output shape are known or inferable from context.

This would seem to suggest a form of functional programming, or treating the overall output as a dependency graph (which can be specified from the command line using something like Click) and building up our overall function accordingly.

```
--add_feature <name> <specification>
```

Where we might circumvent the fact that some features require different specifications than others by automatically generating their spec and associated Click CLI form,

```
--Int2Bin 5
--TileCoder 5 4 random_seed=123
```

It occurs to me that what I am describing are essentially S-Expressions, so perhaps thinking in those terms might be better.


## Implementation of the Actual Pipeline

We could try something along the lines of how Theano build graphs of functions from similar specifications, but then we find ourselves in a bit of a morass because while Theano's design is very close to being able to express what we want to express, it's not quite designed for this use case.
Once your function is compiled, everything is fine, but if you're swapping out components (because you're adjusting your representation) it tends to spend a lot of time recompiling things.

There are also issues in terms of how we are to treat bulk streams of data (multiple inputs arriving as an iterable, say); at first I thought that it would be simple to use NumPy-style broadcasting in order to get this to work, but it turns out that this is ultimately unwieldy for various reasons.

