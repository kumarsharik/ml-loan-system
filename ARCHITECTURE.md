# System Architecture

This diagram shows how data, models, and monitoring flow through the system.

```mermaid
flowchart LR

    User[Loan Applicant] --> API[FastAPI /predict]

    API --> Registry[MLflow Model Registry]
    Registry --> Model[Production Model]

    Model --> API
    API --> Decision[Approve / Review / Reject]

    API --> Log[Prediction Log]

    Log --> Monitor[Drift Monitor]
    Monitor -->|No drift| Sleep[Wait]

    Monitor -->|Drift detected| Train[Train Challenger]
    Train --> Registry

    Registry --> Compare[Champion vs Challenger]
    Compare -->|Challenger better| Promote[Promote to Production]
    Compare -->|Champion wins| Keep[Keep Production]

    Promote --> Registry
    Keep --> Registry
