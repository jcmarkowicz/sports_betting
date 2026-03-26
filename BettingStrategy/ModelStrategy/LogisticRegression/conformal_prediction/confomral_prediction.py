import numpy as np 
import pandas as pd 

def conformal_analysis(
    df_test_results,
    y_pred, 
    y_test, 
    p_values_tcp, 
    results,
    open_odds=True
):
    
    # ---- Build dataframe ----
    if open_odds is not True: 
        test_choice_odds1 = np.where(df_test_results['pred_winner']==1, 
                                    df_test_results['dec_close1_red']-1, 
                                    df_test_results['dec_close1_blue']-1)

        test_choice_odds2 = np.where(df_test_results['pred_winner']==1, 
                                    df_test_results['dec_close2_red']-1, 
                                    df_test_results['dec_close2_blue']-1)

        df_tcp = pd.DataFrame({
            "y_pred": y_pred,
            'y_true': y_test,
            'choice_netodds_close1': test_choice_odds1, 
            'choice_netodds_close2': test_choice_odds2, 
            "p_red":  p_values_tcp[:, 1],
            "p_blue": p_values_tcp[:, 0],
            'p_red_model': results['test_pred_proba'],
            'p_blue_model': 1-results['test_pred_proba']
        })
    
    else: 
        test_open_odds = np.where(df_test_results['pred_winner']==1, 
                            df_test_results['dec_open_red']-1, 
                            df_test_results['dec_open_blue']-1)
        
        print(test_open_odds.shape, y_pred.shape, y_test.shape, p_values_tcp.shape, results['test_pred_proba'].shape)

        df_tcp = pd.DataFrame({
            "y_pred": y_pred,
            'y_true': y_test,
            'test_open_odds': test_open_odds, 
            "p_red":  p_values_tcp[:, 1],
            "p_blue": p_values_tcp[:, 0],
            'p_red_model': results['test_pred_proba'],
            'p_blue_model': 1-results['test_pred_proba']
        })
        print(df_tcp.shape)


    # conformal predicted label
    df_tcp["pred_conf"] = df_tcp[["p_red", "p_blue"]].idxmax(axis=1).map({"p_red":1, "p_blue":0})

    # confidence score
    df_tcp["confidence"] = df_tcp[["p_red","p_blue"]].max(axis=1)

    # uncertainty score
    df_tcp["uncertainty"] = df_tcp[["p_red","p_blue"]].min(axis=1)

    # agreement check
    df_tcp["agreement"] = (df_tcp["y_pred"] == df_tcp["pred_conf"]).astype(int)

    # prediction set size
    alpha = 0.1
    df_tcp["set_size"] = (df_tcp["p_red"] >= alpha).astype(int) + \
                         (df_tcp["p_blue"] >= alpha).astype(int)

    # margin
    df_tcp["gap"] = np.abs(df_tcp["p_red"] - df_tcp["p_blue"])

    # ---- PLOTTING ----
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    axes = axes.flatten()

    # hist p_red + p_blue
    axes[0].hist(df_tcp["p_red"], bins=40, alpha=0.6, label="p_red")
    axes[0].hist(df_tcp["p_blue"], bins=40, alpha=0.6, label="p_blue")
    axes[0].legend()
    axes[0].set_title("p-value distributions")

    # scatter p_red vs p_blue
    axes[1].scatter(df_tcp["p_red"], df_tcp["p_blue"], alpha=0.5)
    axes[1].set_xlabel("p_red")
    axes[1].set_ylabel("p_blue")
    axes[1].set_title("Joint p-value scatter")

    # set size histogram
    axes[2].hist(df_tcp["set_size"], bins=[0, 1, 2, 3], align="left", rwidth=0.8)
    axes[2].set_title("Conformal set size distribution")
    axes[2].set_xticks([0, 1, 2])

    # confidence by predicted class
    sns.boxplot(x="y_pred", y="confidence", data=df_tcp, ax=axes[3])
    axes[3].set_title("Confidence by predicted class")

    # margin histogram
    axes[4].hist(df_tcp["gap"], bins=40)
    axes[4].set_title("Conformal margin distribution")

    # agreement score text
    agreement_rate = df_tcp["agreement"].mean()
    axes[5].text(0.1, 0.5, f"Agreement rate = {agreement_rate:.3f}", fontsize=14)
    axes[5].axis("off")

    plt.tight_layout()
    plt.show()

    return df_tcp