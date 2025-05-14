"Why doesn't it work like this: python complex_mlp/train.py?"


Because:
When you run train.py directly like this, Python treats it as a top-level script, meaning it's not part of a package.
Therefore, it cannot resolve the from complex_mlp... import statement, since it has no knowledge of where the complex_mlp "root" is.

```
**python -m complex_mlp.train**
```

This tells Python:
"complex_mlp.train is a package module, don't try to run it as a plain script."



Ha emlékszel még ennek a beszélgetésnek a teljes tartalmára úgy kezdtük hogy egy képet daraboltunk fel (most 16x16 px es kockákra ) és abból tanítottuk be a complex_mlp t
