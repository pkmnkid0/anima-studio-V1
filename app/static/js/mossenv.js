// Backend tab: the MOSS-Image trainer card (the sd-scripts fork) - environment install and verify.
const MossEnv = EnvCard.create({
  prefix: "moss", name: "MOSS-Image", setupLabel: "Set up MOSS-Image environment", repairLabel: "Repair / update the environment",
  missingText: "The bundled MOSS-Image trainer is missing - redownload Anima Studio.",
  api: { status: () => Api.mossSetupStatus(), start: () => Api.mossSetupStart(), cancel: () => Api.mossSetupCancel(), verify: () => Api.mossVerify() },
});
