import streamlit as st
import numpy as np

def logistic_regression(x):
    return 1 / (1 + np.exp(-x))

def get_user_input():
    user_input = {}
    st.sidebar.header("Model Parameters")
    
    with st.sidebar.expander("Credit History"):
        user_input['ExternalRiskEstimate'] = st.slider('External Risk Estimate', 0, 100, 50)
        user_input['MSinceOldestTradeOpen'] = st.slider('Months Since Oldest Trade Open', 0, 850, 100)
        user_input['AverageMinFile'] = st.slider('Average Months In File', 0, 400, 60)

    with st.sidebar.expander("Credit Frequency"):
        user_input['MSinceMostRecentTradeOpen'] = st.slider('Months Since Most Recent Trade Open', 0, 400, 10)
        user_input['NumTradesOpeninLast12M'] = st.slider('Number of Trades Open in Last 12 Months', 0, 20, 5)
        user_input['NumInqLast6M'] = st.slider('Number of Inquiries Last 6 Months', 0, 70, 2)
        user_input['NumInqLast6Mexcl7days'] = st.slider('Number of Inquiries Last 6 Months excluding last 7 days', 0, 70, 2)

    with st.sidebar.expander("Negative Activity"):
        user_input['NumTrades60Ever2DerogPubRec'] = st.slider('Number of Trades 60+ Ever Derogatory/Public Records', 0, 20, 1)
        user_input['NumTrades90Ever2DerogPubRec'] = st.slider('Number of Trades 90+ Ever Derogatory/Public Records', 0, 20, 1)
        user_input['MSinceMostRecentDelq'] = st.slider('Months Since Most Recent Delinquency', 0, 100, 30)
        user_input['MaxDelq2PublicRecLast12M'] = st.slider('Max Delinquency in Public Records Last 12 Months', 0, 12, 0, step=1)
        user_input['MaxDelqEver'] = st.slider('Max Delinquency Ever', 1, 10, 1, step=1)

    with st.sidebar.expander("Usage"):
        user_input['NetFractionRevolvingBurden'] = st.slider('Net Fraction Revolving Burden', 0, 250, 50)
        user_input['NetFractionInstallBurden'] = st.slider('Net Fraction Installment Burden', 0, 500, 50)
        user_input['NumRevolvingTradesWBalance'] = st.slider('Number of Revolving Trades with Balance', 0, 35, 5)
        user_input['NumInstallTradesWBalance'] = st.slider('Number of Installment Trades with Balance', 0, 25, 5)
        user_input['NumBank2NatlTradesWHighUtilization'] = st.slider('Number of Bank/National Trades with High Utilization', 0, 20, 2)
        user_input['PercentTradesWBalance'] = st.slider('Percent of Trades with Balance', 0, 100, 50)

    with st.sidebar.expander("Stability"):
        user_input['NumSatisfactoryTrades'] = st.slider('Number of Satisfactory Trades', 0, 80, 20)
        user_input['NumTotalTrades'] = st.slider('Total Number of Trades', 0, 110, 20)
        user_input['PercentTradesNeverDelq'] = st.slider('Percent of Trades Never Delinquent', 0, 100, 20)
        user_input['PercentInstallTrades'] = st.slider('Percent of Installment Trades', 0, 100, 20)

    if st.sidebar.button('Run Model'):
        return user_input
    else:
        return None

def main():
    st.title("Credit Risk Prediction Model")
    user_input = get_user_input()

    if user_input:
        pred_creditHistory = logistic_regression(8.2710 - 0.1033 * user_input['ExternalRiskEstimate'] - 0.0014 * user_input['MSinceOldestTradeOpen'] - 0.0060 * user_input['AverageMinFile'])
        pred_creditFrequency = logistic_regression(-0.2177 - 0.0019 * user_input['MSinceMostRecentTradeOpen'] + 0.0366 * user_input['NumTradesOpeninLast12M'] + 0.3950 * user_input['NumInqLast6M'] - 0.2311 * user_input['NumInqLast6Mexcl7days'])
        pred_negActivity = logistic_regression(0.2638 + 0.1593 * user_input['NumTrades60Ever2DerogPubRec'] + 0.0276 * user_input['NumTrades90Ever2DerogPubRec'] - 0.0109 * user_input['MSinceMostRecentDelq'])
        usage = logistic_regression(-1.4799 + 0.0240 * user_input['NetFractionRevolvingBurden'] + 0.0024 * user_input['NetFractionInstallBurden'] + 0.0087 * user_input['NumRevolvingTradesWBalance'] - 0.0175 * user_input['NumInstallTradesWBalance'] - 0.0974 * user_input['NumBank2NatlTradesWHighUtilization'] + 0.0108 * user_input['PercentTradesWBalance'])
        stability = logistic_regression(4.7294 - 0.0119 * user_input['NumSatisfactoryTrades'] - 0.0008 * user_input['NumTotalTrades'] - 0.0519 * user_input['PercentTradesNeverDelq'] + 0.0131 * user_input['PercentInstallTrades'])

        theme_percentages = [
            pred_creditHistory * 100, 
            pred_creditFrequency * 100, 
            pred_negActivity * 100, 
            usage * 100, 
            stability * 100
        ]

        final_prediction = logistic_regression(-4.6816 + 3.1742 * pred_creditHistory + 2.1591 * pred_creditFrequency + 0.3973 * pred_negActivity + 1.6336 * usage + 1.8478 * stability)
        final_percentage = final_prediction * 100
        
        st.subheader("Results")
        tab1, tab2, tab3 = st.tabs(["Subscale Features", "Final Prediction", "Recommendation"])

        with tab1:
            st.subheader("Subscale Feature Percentages")
            theme_labels = ["Credit History", "Credit Frequency", "Negative Activity", "Usage", "Stability"]
            for label, percentage in zip(theme_labels, theme_percentages):
                st.metric(label, f"{percentage:.2f}%")

        with tab2:
            st.subheader("Final Percentage Output")
            st.metric("Final Prediction", f"{final_percentage:.2f}%")

        with tab3:
            st.subheader("Recommendation")
            if final_percentage >= 50:
                st.error("Reject")
                st.write(f"Probability of Default: {final_percentage:.2f}%")
            else:
                st.success("Accept")
                st.write(f"Probability of Non-Default: {100 - final_percentage:.2f}%")

if __name__ == "__main__":
    main()
