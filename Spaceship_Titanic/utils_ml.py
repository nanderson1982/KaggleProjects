# Functions used in Machine Learning notebooks
# Separating them from the notebooks to make those notebooks more clear and concise

def get_classification_metrics(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Display accuracy, log-loss, classification report, confusion matrix, and the Receiver Operating Characteristic (ROC)) curve"""
    
    # Import statements
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import (log_loss, classification_report, confusion_matrix, roc_curve, 
                                 roc_auc_score, matthews_corrcoef, ConfusionMatrixDisplay)
    
    # Get the train and test accuracy scores
    print(f">> Training - Accuracy: {model.score(X_train, y_train) * 100:.2f}%")
    print(f">> Validation - Accuracy: {model.score(X_val, y_val) * 100:.2f}%")
    print(f">> Testing - Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

    # Get the train and test logloss results
    print(f">> Training - LogLoss: {round(log_loss(y_train, model.predict_proba(X_train)), 4)}")
    print(f">> Validation - LogLoss: {round(log_loss(y_val, model.predict_proba(X_val)), 4)}")
    print(f">> Testing - LogLoss: {round(log_loss(y_test, model.predict_proba(X_test)), 4)}")
    
    # Evaluate model on test data
    y_pred = model.predict(X_test)
    
    # Calculate the MCC for testing data
    mcc = matthews_corrcoef(y_test, y_pred)
    print()
    print(f">> Testing - MCC: {mcc}")
    print()    

    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    classification_report_df = pd.DataFrame(report).transpose()
    print(classification_report(y_test, y_pred))

    # Customize the report
    classification_report_df = classification_report_df.drop('support', axis=1)
    classification_report_df = classification_report_df.round(decimals=2)

    # Create a heatmap of the scores
    ax = sns.heatmap(classification_report_df, annot=True, cmap='Blues', fmt='g')

    # Add labels and title
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Classes')
    ax.set_title('Classification Report')
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix
    cm_display = ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    plt.show()

    # Compute ROC curve and AUC score
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.show()
    


# Define function to select optimizer and learning rate
def select_optimizer(optimize, learn):
    """Used to select an optimizer and learning rate."""
    import tensorflow as tf
    
    optimizer_dict = {'Adam': tf.optimizers.Adam(learning_rate = learn),
                      'SGD': tf.optimizers.SGD(learning_rate = learn),
                      'Adadelta': tf.optimizers.Adadelta(learning_rate = learn),
                      'RMSprop': tf.optimizers.RMSprop(learning_rate = learn),
                      'Adagrad': tf.optimizers.Adagrad(learning_rate = learn),
                      'Adamax': tf.optimizers.Adamax(learning_rate = learn),
                      'Nadam': tf.optimizers.Nadam(learning_rate = learn),
                      'Ftrl': tf.optimizers.Ftrl(learning_rate = learn),
                      }
    
    return optimizer_dict[optimize]



def save_model_results(valve_type: str, model, features_name: str, labels_name: str, X_train, y_train, X_test, y_test, y_pred, opt=None, lr=None):
    """Function to save model results by valve type."""
    
    try:
        # Import libraries
        from sklearn.metrics import (log_loss, matthews_corrcoef, precision_score, recall_score, f1_score,
                                    confusion_matrix, roc_auc_score, cohen_kappa_score, fbeta_score,)
        import os.path as path
        from tensorflow import keras
        from contextlib import redirect_stdout, redirect_stderr
        import pandas as pd
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        
        # Add a small constant value to prevent log loss issues
        epsilon = 1e-15  
        
        # Calculate specificity and sensitivity
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = round(tn / (tn + fp), 4)
        sensitivity = round(tp / (tp + fn), 4)
        
        # Adjust features and labels to be a string to be saved
        #vars_dict = globals()
        #vars_dict = {}
        #features_name = [key for key, value in vars_dict.items() if value is features][0]
        #labels_name = [key for key, value in vars_dict.items() if value is labels][0]

        # Create separate result dataframes based on valve type
        vv_link = path.expanduser('/Users/nathananderson/Documents/College/Ace/git/Antec-Controls/Machine Learning/Results/vv_results.pkl')
        vfx_link = path.expanduser('/Users/nathananderson/Documents/College/Ace/git/Antec-Controls/Machine Learning/Results/vfx_results.pkl')
        cv_link = path.expanduser('/Users/nathananderson/Documents/College/Ace/git/Antec-Controls/Machine Learning/Results/cv_results.pkl')
        
        if valve_type.upper() == "VV":
            if path.exists(vv_link):
                vv_results = pd.read_pickle('/Users/nathananderson/Documents/College/Ace/git/Antec-Controls/Machine Learning/Results/vv_results.pkl')
            else:
                vv_results = pd.DataFrame()
        elif valve_type.upper() == "VFX":
            if path.exists(vfx_link):
                vfx_results = pd.read_pickle('/Users/nathananderson/Documents/College/Ace/git/Antec-Controls/Machine Learning/Results/vfx_results.pkl')
            else:
                vfx_results = pd.DataFrame()
        elif valve_type.upper() == "CV":
            if path.exists(cv_link):
                cv_results = pd.read_pickle('/Users/nathananderson/Documents/College/Ace/git/Antec-Controls/Machine Learning/Results/cv_results.pkl')
            else:
                cv_results = pd.DataFrame()
        
        # Check if model is Logistic Classification
        if isinstance(model, LogisticRegression):
            new_row = {'Model': type(model),
                    'Features DF': features_name,
                    'Labels DF': labels_name,
                    'Random State': model.random_state,
                    'Max Iterations': model.max_iter,
                    'Solver': model.solver,
                    'Penalty': model.penalty,
                    'Class Weight': model.class_weight,
                    'C': model.C,
                    'Training - Accuracy': f"{model.score(X_train, y_train) * 100:.2f}%",
                    'Testing - Accuracy': f"{model.score(X_test, y_test) * 100:.2f}%",
                    'Training - LogLoss': f"{round(log_loss(y_train, model.predict_proba(X_train) + epsilon), 4)}",
                    'Testing - LogLoss': f"{round(log_loss(y_test, model.predict_proba(X_test) + epsilon), 4)}",
                    'Testing - MCC': matthews_corrcoef(y_test, y_pred),
                    'Precision': round(precision_score(y_test, y_pred), 4),
                    'Recall': round(recall_score(y_test, y_pred), 4),
                    'F1 score': round(f1_score(y_test, y_pred), 4),
                    'F2 score': round(fbeta_score(y_test, y_pred, beta=2), 4),
                    'Specificity': specificity,
                    'Sensitivity': sensitivity,
                    'Balanced accuracy': round((sensitivity + specificity) / 2, 4),
                    'Cohen\'s Kappa': round(cohen_kappa_score(y_test, y_pred), 4),
                    'AUC-ROC': round(roc_auc_score(y_test, model.decision_function(X_test)), 4)
            }
        
        # Check if model is Random Forest
        elif isinstance(model, RandomForestClassifier):    
            new_row = {'Model': type(model),
                    'Features DF': features_name,
                    'Labels DF': labels_name,
                    'Random State': model.random_state,
                    'Number of Estimators': model.n_estimators,
                    'Criterion': model.criterion,
                    'Max Depth': model.max_depth,
                    'Min Samples Split': model.min_samples_split,
                    'Min Samples Leaf': model.min_samples_leaf,
                    'Min Weight Fraction Leaf': model.min_weight_fraction_leaf,
                    'Max Features': model.max_features,
                    'Max Leaf Nodes': model.max_leaf_nodes,
                    'Class Weight': model.class_weight,
                    'Training - Accuracy': f"{model.score(X_train, y_train) * 100:.2f}%",
                    'Testing - Accuracy': f"{model.score(X_test, y_test) * 100:.2f}%",
                    'Training - LogLoss': f"{round(log_loss(y_train, model.predict_proba(X_train) + epsilon), 4)}",
                    'Testing - LogLoss': f"{round(log_loss(y_test, model.predict_proba(X_test) + epsilon), 4)}",
                    'Testing - MCC': matthews_corrcoef(y_test, y_pred),
                    'Precision': round(precision_score(y_test, y_pred), 4),
                    'Recall': round(recall_score(y_test, y_pred), 4),
                    'F1 score': round(f1_score(y_test, y_pred), 4),
                    'F2 score': round(fbeta_score(y_test, y_pred, beta=2), 4),
                    'Specificity': specificity,
                    'Sensitivity': sensitivity,
                    'Balanced accuracy': round((sensitivity + specificity) / 2, 4),
                    'Cohen\'s Kappa': round(cohen_kappa_score(y_test, y_pred), 4),
                    'AUC-ROC': round(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]), 4)
            }
        
        # Check if model is a neural network    
        elif isinstance(model, keras.models.Sequential):
            with redirect_stdout(None), redirect_stderr(None):
                loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
                train_loss = round(loss + epsilon, 4)
                train_acc = f"{accuracy * 100:.2f}%"
                
                loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                test_loss = round(loss + epsilon, 4)
                test_acc = f"{accuracy * 100:.2f}%"
                
            # Get the number of layers and their types with dropout rates (if applicable)
            layers = []
            for layer in model.layers:
                layer_info = layer.__class__.__name__
                if hasattr(layer, 'units'):
                    layer_info += f"({layer.units})"
                elif isinstance(layer, keras.layers.Dropout):
                    layer_info += f"({layer.rate:.2f})"
                layers.append(layer_info)

            layers_str = ', '.join(layers)

            new_row = {'Model': 'Neural Network',
                    'Features DF': features_name,
                    'Labels DF': labels_name,
                    'Training - Accuracy': train_acc,
                    'Testing - Accuracy': test_acc,
                    'Training - LogLoss': train_loss,
                    'Testing - LogLoss': test_loss,
                    'Testing - MCC': matthews_corrcoef(y_test, y_pred),
                    'Precision': round(precision_score(y_test, y_pred), 4),
                    'Recall': round(recall_score(y_test, y_pred), 4),
                    'F1 score': round(f1_score(y_test, y_pred), 4),
                    'F2 score': round(fbeta_score(y_test, y_pred, beta=2), 4),
                    'Specificity': specificity,
                    'Sensitivity': sensitivity,
                    'Balanced accuracy': round((sensitivity + specificity) / 2, 4),
                    'Cohen\'s Kappa': round(cohen_kappa_score(y_test, y_pred), 4),
                    'AUC-ROC': round(roc_auc_score(y_test, (model.predict(X_test) > 0.5).astype("int32") + epsilon), 4),
                    'Layers': layers_str,
                    'Optimizer': opt,
                    'Learning Rate': lr
            }
                
        else:
            raise ValueError(f"Unsupported model type: {type(model).__name__}. Supported models are LogisticRegression, RandomForestClassifier, and keras.models.Sequential.")
            
        # Add the new row to the specific valve results dataframe
        valve_type = valve_type.upper()
        if valve_type == "VV":
            vv_results = vv_results.append(new_row, ignore_index = True)
            vv_results.to_pickle('/Users/nathananderson/Documents/College/Ace/git/Antec-Controls/Machine Learning/Results/vv_results.pkl')
            return vv_results

        elif valve_type == "VFX":
            vfx_results = vfx_results.append(new_row, ignore_index = True)
            vfx_results.to_pickle('/Users/nathananderson/Documents/College/Ace/git/Antec-Controls/Machine Learning/Results/vfx_results.pkl')
            return vfx_results
            
        elif valve_type == "CV":
            cv_results = cv_results.append(new_row, ignore_index = True)
            cv_results.to_pickle('/Users/nathananderson/Documents/College/Ace/git/Antec-Controls/Machine Learning/Results/vfx_results.pkl')
            return vv_results
        
        else:
            print('Invalid vavle type.')
    
    except TypeError:
        print('Type Error')