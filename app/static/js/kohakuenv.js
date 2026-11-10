// Backend tab: the Low-VRAM trainer card - environment install and verify.
const KohakuEnv = EnvCard.create({
  prefix: "kohaku", name: "Low-VRAM trainer", setupLabel: "Set up Low-VRAM trainer environment", repairLabel: "Repair / update the environment",
  missingText: "The bundled Low-VRAM trainer (kohaku_backend) is missing - redownload Anima Studio.",
  api: { status: () => Api.kohakuSetupStatus(), start: () => Api.kohakuSetupStart(), cancel: () => Api.kohakuSetupCancel(), verify: () => Api.kohakuVerify() },
});
