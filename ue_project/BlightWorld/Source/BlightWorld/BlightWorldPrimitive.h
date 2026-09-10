#pragma once

#include "CoreMinimal.h"
#include "GameFramework/Actor.h"
#include "BlightWorldPrimitive.generated.h"

class UStaticMesh;
class UStaticMeshComponent;

UENUM(BlueprintType)
enum class EBlightWorldPrimitiveShape : uint8
{
    Box,
    Sphere,
    Cylinder,
    Plane
};

/** Editable MuJoCo primitive metadata with a visual-only Unreal preview. */
UCLASS(BlueprintType, Blueprintable)
class BLIGHTWORLD_API ABlightWorldPrimitive : public AActor
{
    GENERATED_BODY()

public:
    ABlightWorldPrimitive();
    void OnConstruction(const FTransform& Transform) override;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Blight|Physics")
    FName PhysicsName = NAME_None;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Blight|Shape")
    EBlightWorldPrimitiveShape Shape = EBlightWorldPrimitiveShape::Box;

    /** Box half extents in metres; a plane uses X/Y for its visible half extents. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Blight|Shape", meta = (ClampMin = "0.0", Units = "m", EditCondition = "Shape == EBlightWorldPrimitiveShape::Box || Shape == EBlightWorldPrimitiveShape::Plane"))
    FVector HalfSizeM = FVector(0.1, 0.1, 0.1);

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Blight|Shape", meta = (ClampMin = "0.0", Units = "m", EditCondition = "Shape == EBlightWorldPrimitiveShape::Sphere || Shape == EBlightWorldPrimitiveShape::Cylinder"))
    double RadiusM = 0.1;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Blight|Shape", meta = (ClampMin = "0.0", Units = "m", EditCondition = "Shape == EBlightWorldPrimitiveShape::Cylinder"))
    double HalfHeightM = 0.1;

    /** Zero exports a static primitive; positive mass is simulated only in MuJoCo. */
    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Blight|Physics", meta = (ClampMin = "0.0", Units = "kg"))
    double MassKg = 0.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Blight|Physics", meta = (ClampMin = "0.0"))
    double SlidingFriction = 1.0;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Blight|Physics", meta = (ClampMin = "0.0"))
    double TorsionalFriction = 0.005;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Blight|Physics", meta = (ClampMin = "0.0"))
    double RollingFriction = 0.0001;

    UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Blight|Physics")
    FName MaterialName = NAME_None;

    UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Blight|Preview")
    TObjectPtr<UStaticMeshComponent> PreviewMesh;

private:
    void UpdatePreview();

    UPROPERTY()
    TObjectPtr<UStaticMesh> BoxMesh;

    UPROPERTY()
    TObjectPtr<UStaticMesh> SphereMesh;

    UPROPERTY()
    TObjectPtr<UStaticMesh> CylinderMesh;

    UPROPERTY()
    TObjectPtr<UStaticMesh> PlaneMesh;
};
