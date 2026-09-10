# luma-ml

Classic machine learning algorithms built on `luma-tensor`.

## Algorithms

- **Linear models**: `LinearRegression`, `RidgeRegression`, `LassoRegression`, `LogisticRegression`
- **Tree & ensemble**: `DecisionTreeClassifier` / `DecisionTreeRegressor`, `RandomForestClassifier`, `AdaBoostRegressor`, `GradientBoostRegressor`
- **Neighbors**: `KnnClassifier`, `KnnRegression`
- **Naive Bayes**: `GaussianNB`, `MultinomialNB`
- **Cluster**: `KMeans`, `DBSCAN`
- **Pipeline**: `Pipeline` + `pipelines!` macro
- **Metrics**: confusion matrix, accuracy / precision / recall / f1, MSE / MAE / R2
- **Datasets**: built-in iris & diabetes loaders, `train_test_split`, `make_regression`

## Usage

```rust
use luma_tensor::Cpu;
use luma_ml::{PredictFit, PredictModel};
use luma_ml::datasets::{load_iris, train_test_split};
use luma_ml::metrics::accuracy_score;
use luma_ml::tree::DecisionTreeClassifier;

let iris = load_iris(&Cpu).unwrap();
let (x_train, x_test, y_train, y_test) = train_test_split(&iris.data, &iris.target, 0.3).unwrap();

let model = DecisionTreeClassifier::new(10).fit(&x_train, &y_train).unwrap();
let y_pred = model.predict(&x_test).unwrap();
let acc = accuracy_score(&y_test, &y_pred).unwrap();
println!("accuracy: {acc}");
```

## License

MIT — see [LICENSE](LICENSE).
